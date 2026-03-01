"""
Paper trading executor with persistent SQLite state.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from loguru import logger

from config.settings import BASE_DIR, XRP_ONLY_SYMBOL, settings
from execution.base_executor import BaseExecutor
from risk.safety_layer import validate_position_size, validate_price


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_paper_snapshot(
    db_path: Optional[str] = None,
    default_balance: Optional[float] = None,
) -> Dict[str, Any]:
    """Read persisted paper state without creating an executor instance."""
    path = _resolve_paper_db_path(db_path)
    fallback_balance = (
        settings.trading.paper_starting_balance
        if default_balance is None
        else float(default_balance)
    )
    if not path.exists():
        return {
            "db_path": str(path),
            "balance": fallback_balance,
            "realized_pnl": 0.0,
            "positions": {},
            "trade_count": 0,
            "starting_balance": fallback_balance,
        }

    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000;")
    try:
        starting_balance_row = conn.execute(
            "SELECT value FROM state WHERE key = 'starting_balance'"
        ).fetchone()
        starting_balance = (
            _safe_float(starting_balance_row["value"]) if starting_balance_row else fallback_balance
        )

        realized_row = conn.execute(
            "SELECT COALESCE(SUM(COALESCE(realized_pnl, 0.0)), 0.0) AS pnl FROM trades"
        ).fetchone()
        cash_delta_row = conn.execute(
            """
            SELECT COALESCE(SUM(
                CASE
                    WHEN UPPER(side) = 'BUY' THEN -(fill_price * quantity + fee)
                    WHEN UPPER(side) = 'SELL' THEN  (fill_price * quantity - fee)
                    ELSE 0.0
                END
            ), 0.0) AS cash_delta
            FROM trades
            """
        ).fetchone()

        positions_rows = conn.execute(
            "SELECT symbol, quantity, avg_entry_price, updated_at FROM positions"
        ).fetchall()
        trade_count_row = conn.execute("SELECT COUNT(*) AS c FROM trades").fetchone()

        positions = {
            row["symbol"]: {
                "symbol": row["symbol"],
                "quantity": _safe_float(row["quantity"]),
                "avg_entry_price": _safe_float(row["avg_entry_price"]),
                "updated_at": row["updated_at"],
            }
            for row in positions_rows
        }

        return {
            "db_path": str(path),
            # IMPORTANT: derive from trades + starting_balance to avoid trusting mutable state rows.
            "balance": float(starting_balance)
            + (_safe_float(cash_delta_row["cash_delta"]) if cash_delta_row else 0.0),
            "realized_pnl": _safe_float(realized_row["pnl"]) if realized_row else 0.0,
            "positions": positions,
            "trade_count": int(trade_count_row["c"]) if trade_count_row else 0,
            "starting_balance": float(starting_balance),
        }
    finally:
        conn.close()


def _resolve_paper_db_path(db_path: Optional[str] = None) -> Path:
    candidate = Path(db_path or settings.trading.paper_db_path)
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = BASE_DIR / "data" / "paper_trading.db"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(
            f"Cannot use PAPER_DB_PATH={candidate}. Falling back to {fallback}."
        )
        return fallback


class PaperExecutor(BaseExecutor):
    """Executor that simulates fills and account state with local persistence."""

    mode = "PAPER"

    def __init__(
        self,
        fetcher,
        starting_balance: Optional[float] = None,
        fee_rate: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        db_path: Optional[str] = None,
    ) -> None:
        super().__init__(fetcher)
        self._lock = threading.RLock()
        self.db_path = _resolve_paper_db_path(db_path)
        self.fee_rate = settings.trading.paper_fee_rate if fee_rate is None else fee_rate
        self.slippage_bps = settings.trading.paper_slippage_bps if slippage_bps is None else slippage_bps
        self._default_balance = (
            settings.trading.paper_starting_balance
            if starting_balance is None
            else float(starting_balance)
        )

        self._balance: float = self._default_balance
        self._realized_pnl: float = 0.0
        # Position fields:
        # - quantity: base units (positive = long, negative = short)
        # - avg_entry_price: raw fill price (excludes fees)
        # - entry_fee_total: accumulated entry fees paid for the open quantity
        self._positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self._last_prices: Dict[str, float] = {}

        self._init_db()
        self._load_state()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    entry_fee_total REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    side TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    requested_price REAL,
                    fill_price REAL NOT NULL,
                    fee REAL NOT NULL,
                    realized_pnl REAL,
                    balance_after REAL NOT NULL
                )
                """
            )

            # Lightweight migrations for existing sqlite snapshots.
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(positions)").fetchall()}
            if "entry_fee_total" not in cols:
                conn.execute(
                    "ALTER TABLE positions ADD COLUMN entry_fee_total REAL NOT NULL DEFAULT 0.0"
                )

            # Persist an immutable starting balance baseline (used for derived metrics).
            row = conn.execute(
                "SELECT value FROM state WHERE key = 'starting_balance'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT OR REPLACE INTO state (key, value) VALUES ('starting_balance', ?)",
                    (float(self._default_balance),),
                )
            conn.commit()
        finally:
            conn.close()

    def _load_state(self) -> None:
        """
        Load sqlite snapshot, but derive balance/realized_pnl from the trades table.

        This makes "balance" robust against corrupted mutable state rows (e.g., a past bug
        or manual edits) and keeps metrics consistent with DB truth.
        """
        conn = self._connect()
        try:
            starting_balance_row = conn.execute(
                "SELECT value FROM state WHERE key = 'starting_balance'"
            ).fetchone()
            starting_balance = (
                _safe_float(starting_balance_row["value"]) if starting_balance_row else self._default_balance
            )

            pos_rows = conn.execute(
                "SELECT symbol, quantity, avg_entry_price, entry_fee_total, updated_at FROM positions"
            ).fetchall()
            self._positions = {
                row["symbol"]: {
                    "quantity": _safe_float(row["quantity"]),
                    "avg_entry_price": _safe_float(row["avg_entry_price"]),
                    "entry_fee_total": _safe_float(row["entry_fee_total"]),
                }
                for row in pos_rows
            }

            rows = conn.execute(
                """
                SELECT timestamp, side, symbol, quantity, requested_price,
                       fill_price, fee, realized_pnl, balance_after
                FROM trades
                ORDER BY id ASC
                """
            ).fetchall()
            self.trade_history = [
                {
                    "timestamp": row["timestamp"],
                    "side": row["side"],
                    "symbol": row["symbol"],
                    "quantity": _safe_float(row["quantity"]),
                    "requested_price": _safe_float(row["requested_price"], default=None),
                    "fill_price": _safe_float(row["fill_price"]),
                    "fee": _safe_float(row["fee"]),
                    "realized_pnl": _safe_float(row["realized_pnl"], default=None),
                    "balance_after": _safe_float(row["balance_after"]),
                }
                for row in rows
            ]

            trade_count = int(conn.execute("SELECT COUNT(*) AS c FROM trades").fetchone()["c"] or 0)
            if trade_count == 0 and self._positions:
                logger.bind(
                    event="PAPER_STATE_CORRUPT",
                    reason="positions_without_trades",
                    db_path=str(self.db_path),
                    positions=list(self._positions.keys()),
                ).critical("PAPER_STATE_CORRUPT")
                self._positions.clear()
                conn.execute("DELETE FROM positions")
                conn.commit()

            realized_row = conn.execute(
                "SELECT COALESCE(SUM(COALESCE(realized_pnl, 0.0)), 0.0) AS pnl FROM trades"
            ).fetchone()
            cash_delta_row = conn.execute(
                """
                SELECT COALESCE(SUM(
                    CASE
                        WHEN UPPER(side) = 'BUY' THEN -(fill_price * quantity + fee)
                        WHEN UPPER(side) = 'SELL' THEN  (fill_price * quantity - fee)
                        ELSE 0.0
                    END
                ), 0.0) AS cash_delta
                FROM trades
                """
            ).fetchone()

            self._realized_pnl = _safe_float(realized_row["pnl"]) if realized_row else 0.0
            cash_delta = _safe_float(cash_delta_row["cash_delta"]) if cash_delta_row else 0.0
            self._balance = float(starting_balance) + float(cash_delta)
        finally:
            conn.close()

    def _persist_trade_and_state(
        self,
        *,
        side: str,
        symbol: str,
        qty: float,
        requested_price: Optional[float],
        fill_price: float,
        fee: float,
        realized_pnl: Optional[float],
    ) -> None:
        event: Dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "side": side,
            "symbol": symbol,
            "quantity": float(qty),
            "requested_price": requested_price,
            "fill_price": float(fill_price),
            "fee": float(fee),
            "realized_pnl": realized_pnl,
            "balance_after": float(self._balance),
        }

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO trades (
                    timestamp, side, symbol, quantity, requested_price,
                    fill_price, fee, realized_pnl, balance_after
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["timestamp"],
                    event["side"],
                    event["symbol"],
                    event["quantity"],
                    event["requested_price"],
                    event["fill_price"],
                    event["fee"],
                    event["realized_pnl"],
                    event["balance_after"],
                ),
            )

            # Persist numeric snapshot for observability only (derived values are source of truth).
            conn.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES ('balance', ?)",
                (float(self._balance),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES ('realized_pnl', ?)",
                (float(self._realized_pnl),),
            )

            conn.execute("DELETE FROM positions")
            for pos_symbol, position in self._positions.items():
                conn.execute(
                    """
                    INSERT INTO positions (symbol, quantity, avg_entry_price, entry_fee_total, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        pos_symbol,
                        float(position["quantity"]),
                        float(position["avg_entry_price"]),
                        float(position.get("entry_fee_total", 0.0)),
                        datetime.now(UTC).isoformat(),
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        self.trade_history.append(event)

    def _allocate_entry_fee(self, symbol: str, qty_to_close: float) -> float:
        position = self._positions.get(symbol)
        if not position:
            return 0.0
        available_qty = abs(float(position.get("quantity", 0.0)))
        if available_qty <= 0 or qty_to_close <= 0:
            return 0.0
        entry_fee_total = float(position.get("entry_fee_total", 0.0))
        if entry_fee_total <= 0:
            return 0.0
        ratio = min(1.0, float(qty_to_close) / float(available_qty))
        allocated = entry_fee_total * ratio
        position["entry_fee_total"] = max(0.0, entry_fee_total - allocated)
        return allocated

    def _market_price(
        self,
        symbol: str,
        side: str,
        requested_price: Optional[float],
        ticker: Optional[Dict[str, Any]] = None,
    ) -> float:
        if symbol != XRP_ONLY_SYMBOL:
            raise RuntimeError(f"XRP-only mode: unsupported symbol {symbol}")
        if requested_price is not None and requested_price > 0:
            return float(requested_price)
        else:
            if ticker is None:
                ticker = self.fetcher.get_ticker(symbol)
            if side == "buy":
                reference_price = _safe_float(
                    ticker.get("best_ask"), _safe_float(ticker.get("price"))
                )
            else:
                reference_price = _safe_float(
                    ticker.get("best_bid"), _safe_float(ticker.get("price"))
                )

        slippage = self.slippage_bps / 10_000.0
        if side == "buy":
            return reference_price * (1 + slippage)
        return reference_price * (1 - slippage)

    def get_balance(self) -> float:
        return float(self._balance)

    def get_positions(self) -> Dict[str, Any]:
        return {
            symbol: {
                "symbol": symbol,
                "quantity": float(position["quantity"]),
                "avg_entry_price": float(position["avg_entry_price"]),
                "side": "LONG" if float(position["quantity"]) > 0 else "SHORT",
            }
            for symbol, position in self._positions.items()
            if abs(float(position["quantity"])) > 1e-12
        }

    def get_realized_pnl(self) -> float:
        return float(self._realized_pnl)

    def get_unrealized_pnl(self) -> float:
        unrealized = 0.0
        account_balance = max(1.0, self._balance)
        for symbol, position in list(self._positions.items()):
            qty = float(position["quantity"])
            entry_price = float(position["avg_entry_price"])
            entry_fee_total = float(position.get("entry_fee_total", 0.0))
            if entry_price <= 0 or abs(qty) <= 1e-12:
                continue
            try:
                market = self.fetcher.get_ticker(symbol)
                mark_price = _safe_float(market.get("price"), entry_price)
            except Exception:
                logger.bind(event="PNL_GUARD", pair=symbol).warning(
                    "Unable to fetch mark price, skipping unrealized update"
                )
                continue
            pnl = (mark_price - entry_price) * qty - entry_fee_total
            if abs(pnl) > account_balance * 5:
                logger.bind(
                    event="PNL_GUARD_TRIGGERED",
                    pair=symbol,
                    unrealized_pnl=float(pnl),
                    threshold=float(account_balance * 5),
                    balance=float(account_balance),
                ).critical("PNL_GUARD_TRIGGERED")
                self.close_position(symbol)
                continue
            unrealized += pnl
        return unrealized

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        unrealized = self.get_unrealized_pnl()
        return {
            "mode": self.mode,
            "balance": self.get_balance(),
            "realized_pnl": self.get_realized_pnl(),
            "unrealized_pnl": unrealized,
            "equity": self.get_balance() + unrealized,
            "positions": self.get_positions(),
            "trade_count": len(self.trade_history),
            "db_path": str(self.db_path),
        }

    def buy(
        self,
        symbol: str,
        qty: float,
        price: Optional[float] = None,
        market_ticker: Optional[Dict[str, Any]] = None,
        ticker: Optional[Dict[str, Any]] = None,  # backward-compatible alias
    ) -> Dict[str, Any]:
        if symbol != XRP_ONLY_SYMBOL:
            raise RuntimeError(f"XRP-only mode: unsupported symbol {symbol}")
        with self._lock:
            try:
                resolved_ticker = market_ticker or ticker or self.fetcher.get_ticker(symbol)
            except Exception as exc:
                logger.bind(event="VALIDATION_ERROR", pair=symbol, error=str(exc)).error(
                    "Could not fetch ticker"
                )
                raise

            mark_price = _safe_float(resolved_ticker.get("price"), 0.0)
            last_price = self._last_prices.get(symbol)
            if not validate_price(symbol, mark_price, last_price):
                raise ValueError("Invalid mark price")
            self._last_prices[symbol] = float(mark_price)

            fill_price = self._market_price(symbol, "buy", price, ticker=resolved_ticker)
            if not validate_price(symbol, fill_price, last_price or mark_price):
                raise ValueError("Invalid entry price")
            if qty <= 0:
                raise ValueError("Invalid quantity")
            if fill_price > 0:
                price_ratio = abs(mark_price - fill_price) / fill_price
                if price_ratio > 0.5:
                    raise ValueError("Price scale mismatch detected")

            fee = fill_price * qty * self.fee_rate
            notional = fill_price * qty
            total_cost = notional + fee
            if total_cost > self._balance + 1e-12:
                logger.bind(
                    event="PAPER_BALANCE_INSUFFICIENT",
                    mode="paper",
                    side="buy",
                    pair=symbol,
                    qty=float(qty),
                    fill_price=float(fill_price),
                    notional=float(notional),
                    fee=float(fee),
                    required_balance=float(total_cost),
                    available_balance=float(self._balance),
                ).error("Insufficient paper balance for BUY")
                return cast(Dict[str, Any], False)

            position = self._positions.get(
                symbol,
                {"quantity": 0.0, "avg_entry_price": 0.0, "entry_fee_total": 0.0},
            )
            old_qty = float(position["quantity"])
            old_avg = float(position["avg_entry_price"])
            old_entry_fee = float(position.get("entry_fee_total", 0.0))

            realized_pnl: Optional[float] = None

            # Flat or long -> add/increase long.
            if old_qty >= 0:
                if not validate_position_size(symbol, notional, float(self._balance)):
                    raise ValueError("Trade blocked by position limits")
                new_qty = old_qty + qty
                new_avg = ((old_qty * old_avg) + (qty * fill_price)) / new_qty if new_qty > 0 else 0.0
                self._positions[symbol] = {
                    "quantity": new_qty,
                    "avg_entry_price": new_avg,
                    "entry_fee_total": old_entry_fee + float(fee),
                }
            else:
                # Existing short: buy closes part/all, and can flip to long.
                short_abs = abs(old_qty)
                close_qty = min(qty, short_abs)
                open_qty = max(0.0, qty - close_qty)
                fee_close = fee * (close_qty / qty) if qty > 0 else 0.0
                fee_open = max(0.0, fee - fee_close)

                allocated_entry_fee = self._allocate_entry_fee(symbol, close_qty)
                if close_qty > 0:
                    realized_component = (old_avg - fill_price) * close_qty - allocated_entry_fee - fee_close
                    realized_pnl = float(realized_component)
                    self._realized_pnl += float(realized_component)

                remaining_short = max(0.0, short_abs - close_qty)
                if remaining_short > 1e-12:
                    self._positions[symbol] = {
                        "quantity": -remaining_short,
                        "avg_entry_price": old_avg,
                        "entry_fee_total": float(position.get("entry_fee_total", 0.0)),
                    }
                elif open_qty > 1e-12:
                    if not validate_position_size(symbol, fill_price * open_qty, float(self._balance)):
                        raise ValueError("Trade blocked by position limits")
                    self._positions[symbol] = {
                        "quantity": open_qty,
                        "avg_entry_price": fill_price,
                        "entry_fee_total": fee_open,
                    }
                else:
                    self._positions.pop(symbol, None)

            self._balance -= total_cost

            try:
                self._persist_trade_and_state(
                    side="BUY",
                    symbol=symbol,
                    qty=qty,
                    requested_price=price,
                    fill_price=fill_price,
                    fee=fee,
                    realized_pnl=realized_pnl,
                )
            except Exception:
                logger.exception("Failed to persist paper BUY; reloading snapshot")
                self._load_state()
                raise

            logger.bind(
                event="TRADE_EXECUTED",
                mode="paper",
                side="buy",
                pair=symbol,
                price=float(fill_price),
                qty=float(qty),
                fee=float(fee),
                notional=float(notional),
                realized_pnl=(None if realized_pnl is None else float(realized_pnl)),
                balance=float(self._balance),
            ).info("TRADE_EXECUTED")

            return {
                "status": "filled",
                "symbol": symbol,
                "side": "buy",
                "quantity": qty,
                "fill_price": fill_price,
                "fee": fee,
                "realized_pnl": realized_pnl,
                "latency_ms": 0.0,
            }

    def sell(self, symbol: str, qty: float, price: Optional[float] = None) -> Dict[str, Any]:
        if symbol != XRP_ONLY_SYMBOL:
            raise RuntimeError(f"XRP-only mode: unsupported symbol {symbol}")
        try:
            with self._lock:
                try:
                    market_ticker = self.fetcher.get_ticker(symbol)
                except Exception as exc:
                    logger.bind(event="VALIDATION_ERROR", pair=symbol, error=str(exc)).error(
                        "VALIDATION_ERROR"
                    )
                    raise

                mark_price = _safe_float(market_ticker.get("price"), 0.0)
                last_price = self._last_prices.get(symbol)
                if not validate_price(symbol, mark_price, last_price):
                    raise ValueError("Invalid mark price")
                self._last_prices[symbol] = float(mark_price)
                if qty <= 0:
                    raise ValueError("Invalid quantity")

                fill_price = self._market_price(symbol, "sell", price)
                if not validate_price(symbol, fill_price, last_price or mark_price):
                    raise ValueError("Invalid exit price")
                if fill_price > 0:
                    price_ratio = abs(mark_price - fill_price) / fill_price
                    if price_ratio > 0.5:
                        raise ValueError("Price scale mismatch detected")

                fee = fill_price * qty * self.fee_rate
                notional = fill_price * qty
                proceeds = notional - fee
                position = self._positions.get(
                    symbol,
                    {"quantity": 0.0, "avg_entry_price": 0.0, "entry_fee_total": 0.0},
                )
                old_qty = float(position["quantity"])
                old_avg = float(position["avg_entry_price"])
                old_entry_fee = float(position.get("entry_fee_total", 0.0))

                realized_pnl: Optional[float] = None

                # Flat or short -> add/increase short.
                if old_qty <= 0:
                    if not validate_position_size(symbol, notional, float(self._balance)):
                        raise ValueError("Trade blocked by position limits")
                    short_abs = abs(old_qty)
                    new_short_abs = short_abs + qty
                    new_avg = ((short_abs * old_avg) + (qty * fill_price)) / new_short_abs if new_short_abs > 0 else 0.0
                    self._positions[symbol] = {
                        "quantity": -new_short_abs,
                        "avg_entry_price": new_avg,
                        "entry_fee_total": old_entry_fee + float(fee),
                    }
                else:
                    # Existing long: sell closes part/all, and can flip to short.
                    long_qty = old_qty
                    close_qty = min(qty, long_qty)
                    open_qty = max(0.0, qty - close_qty)
                    fee_close = fee * (close_qty / qty) if qty > 0 else 0.0
                    fee_open = max(0.0, fee - fee_close)

                    allocated_entry_fee = self._allocate_entry_fee(symbol, close_qty)
                    if close_qty > 0:
                        realized_component = (fill_price - old_avg) * close_qty - allocated_entry_fee - fee_close
                        realized_pnl = float(realized_component)
                        self._realized_pnl += float(realized_component)

                    remaining_long = max(0.0, long_qty - close_qty)
                    if remaining_long > 1e-12:
                        self._positions[symbol] = {
                            "quantity": remaining_long,
                            "avg_entry_price": old_avg,
                            "entry_fee_total": float(position.get("entry_fee_total", 0.0)),
                        }
                    elif open_qty > 1e-12:
                        if not validate_position_size(symbol, fill_price * open_qty, float(self._balance)):
                            raise ValueError("Trade blocked by position limits")
                        self._positions[symbol] = {
                            "quantity": -open_qty,
                            "avg_entry_price": fill_price,
                            "entry_fee_total": fee_open,
                        }
                    else:
                        self._positions.pop(symbol, None)

                self._balance += proceeds

                try:
                    self._persist_trade_and_state(
                        side="SELL",
                        symbol=symbol,
                        qty=qty,
                        requested_price=price,
                        fill_price=fill_price,
                        fee=fee,
                        realized_pnl=realized_pnl,
                    )
                except Exception:
                    logger.exception("Failed to persist paper SELL; reloading snapshot")
                    self._load_state()
                    raise

                logger.bind(
                    event="TRADE_EXECUTED",
                    mode="paper",
                    side="sell",
                    pair=symbol,
                    price=float(fill_price),
                    qty=float(qty),
                    fee=float(fee),
                    realized_pnl=(None if realized_pnl is None else float(realized_pnl)),
                    balance=float(self._balance),
                ).info("TRADE_EXECUTED")

                return {
                    "status": "filled",
                    "symbol": symbol,
                    "side": "sell",
                    "quantity": qty,
                    "fill_price": fill_price,
                    "fee": fee,
                    "realized_pnl": realized_pnl,
                    "latency_ms": 0.0,
                }
        except ValueError as exc:
            if "Insufficient paper balance" in str(exc):
                logger.bind(
                    event="PAPER_BALANCE_INSUFFICIENT",
                    mode="paper",
                    side="sell",
                    pair=symbol,
                    qty=float(qty),
                    requested_price=(None if price is None else float(price)),
                    available_balance=float(self._balance),
                    error=str(exc),
                ).error("Insufficient paper balance for SELL")
                return cast(Dict[str, Any], False)
            raise

    def close_position(self, symbol: str) -> Dict[str, Any]:
        position = self._positions.get(symbol)
        if not position or abs(float(position["quantity"])) <= 1e-12:
            return {"status": "no_position", "symbol": symbol, "quantity": 0.0}
        quantity = float(position["quantity"])
        if quantity > 0:
            return self.sell(symbol, quantity)
        return self.buy(symbol, abs(quantity))

    def place_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
    ) -> Dict[str, Any]:
        logger.debug(
            f"[PAPER STOP] {symbol} side={side} qty={quantity:.8f} stop={stop_price:.8f}"
        )
        return {
            "status": "simulated",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "stop_price": stop_price,
        }

    def place_limit_sell(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        logger.debug(
            f"[PAPER TARGET] {symbol} qty={quantity:.8f} target={price:.8f}"
        )
        return {
            "status": "simulated",
            "symbol": symbol,
            "side": "sell",
            "quantity": quantity,
            "price": price,
        }

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> Dict[str, Any]:
        logger.debug(
            f"[PAPER LIMIT] {symbol} side={side} qty={quantity:.8f} price={price:.8f}"
        )
        return {
            "status": "simulated",
            "symbol": symbol,
            "side": str(side).lower(),
            "quantity": quantity,
            "price": price,
        }

    def cancel_order(self, order_id: str) -> bool:
        logger.debug(f"[PAPER CANCEL] order={order_id}")
        return True

    def cancel_all_orders(self, symbol: str) -> None:
        logger.debug(f"[PAPER CANCEL_ALL] symbol={symbol}")

    def reconcile_positions(self) -> None:
        with self._lock:
            self._load_state()
            symbols = list(self._positions.keys()) or [XRP_ONLY_SYMBOL]
            events = self.schedule_reconciliation(symbols, source="paper_executor")
            logger.bind(
                reconciliation={
                    "symbol": ",".join(symbols),
                    "event": "reconcile_request",
                    "details": {
                        "mode": "PAPER",
                        "queued_events": len(events),
                    },
                }
            ).info("RECONCILIATION_EVENT")

    def reload_runtime_state(self) -> None:
        with self._lock:
            self._load_state()

    def safe_reset(self, *, initial_balance: Optional[float] = None) -> None:
        """
        Safe paper reset (sqlite only):
        - wipe trades/positions/state
        - reset starting_balance/balance/realized_pnl
        - VACUUM sqlite
        """
        with self._lock:
            target = float(self._default_balance if initial_balance is None else initial_balance)
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM trades")
                conn.execute("DELETE FROM positions")
                conn.execute("DELETE FROM state")
                conn.executemany(
                    "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                    (
                        ("starting_balance", target),
                        ("balance", target),
                        ("realized_pnl", 0.0),
                    ),
                )
                conn.commit()
                conn.execute("VACUUM")
            finally:
                conn.close()

            self._balance = target
            self._realized_pnl = 0.0
            self._positions.clear()
            self.trade_history.clear()
            self._last_prices.clear()
            logger.bind(event="PAPER_SAFE_RESET", balance=target, db_path=str(self.db_path)).critical(
                "PAPER_SAFE_RESET"
            )
