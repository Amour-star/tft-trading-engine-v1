"""
Paper trading executor with persistent SQLite state.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from config.settings import BASE_DIR, settings
from execution.base_executor import BaseExecutor


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
        }

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        balance_row = conn.execute(
            "SELECT value FROM state WHERE key = 'balance'"
        ).fetchone()
        realized_row = conn.execute(
            "SELECT value FROM state WHERE key = 'realized_pnl'"
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
            "balance": _safe_float(balance_row["value"]) if balance_row else fallback_balance,
            "realized_pnl": _safe_float(realized_row["value"]) if realized_row else 0.0,
            "positions": positions,
            "trade_count": int(trade_count_row["c"]) if trade_count_row else 0,
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
        self._positions: Dict[str, Dict[str, float]] = {}
        self.trade_history: List[Dict[str, Any]] = []

        self._init_db()
        self._load_state()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
            conn.commit()
        finally:
            conn.close()

    def _load_state(self) -> None:
        snapshot = load_paper_snapshot(str(self.db_path), default_balance=self._default_balance)
        self._balance = float(snapshot.get("balance", self._default_balance))
        self._realized_pnl = float(snapshot.get("realized_pnl", 0.0))
        self._positions = {
            symbol: {
                "quantity": float(position["quantity"]),
                "avg_entry_price": float(position["avg_entry_price"]),
            }
            for symbol, position in snapshot.get("positions", {}).items()
        }

        conn = self._connect()
        try:
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
        finally:
            conn.close()

    def _persist_state(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES ('balance', ?)",
                (self._balance,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES ('realized_pnl', ?)",
                (self._realized_pnl,),
            )
            conn.execute("DELETE FROM positions")
            for symbol, position in self._positions.items():
                conn.execute(
                    """
                    INSERT INTO positions (symbol, quantity, avg_entry_price, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        float(position["quantity"]),
                        float(position["avg_entry_price"]),
                        datetime.now(UTC).isoformat(),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def _record_trade(
        self,
        side: str,
        symbol: str,
        qty: float,
        requested_price: Optional[float],
        fill_price: float,
        fee: float,
        realized_pnl: Optional[float],
    ) -> None:
        event = {
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
        self.trade_history.append(event)

        conn = self._connect()
        try:
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
            conn.commit()
        finally:
            conn.close()

    def _market_price(self, symbol: str, side: str, requested_price: Optional[float]) -> float:
        if requested_price is not None and requested_price > 0:
            reference_price = requested_price
        else:
            ticker = self.fetcher.get_ticker(symbol)
            if side == "buy":
                reference_price = _safe_float(ticker.get("best_ask"), _safe_float(ticker.get("price")))
            else:
                reference_price = _safe_float(ticker.get("best_bid"), _safe_float(ticker.get("price")))

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
            }
            for symbol, position in self._positions.items()
            if position["quantity"] > 0
        }

    def get_realized_pnl(self) -> float:
        return float(self._realized_pnl)

    def get_unrealized_pnl(self) -> float:
        unrealized = 0.0
        for symbol, position in self._positions.items():
            qty = float(position["quantity"])
            if qty <= 0:
                continue
            try:
                market = self.fetcher.get_ticker(symbol)
                current_price = _safe_float(market.get("price"), float(position["avg_entry_price"]))
            except Exception:
                current_price = float(position["avg_entry_price"])
            unrealized += (current_price - float(position["avg_entry_price"])) * qty
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

    def buy(self, symbol: str, qty: float, price: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            fill_price = self._market_price(symbol, "buy", price)
            fee = fill_price * qty * self.fee_rate
            notional = fill_price * qty
            total_cost = notional + fee

            if total_cost > self._balance + 1e-12:
                raise ValueError(
                    f"Insufficient paper balance. Need {total_cost:.4f}, have {self._balance:.4f}"
                )

            position = self._positions.get(symbol, {"quantity": 0.0, "avg_entry_price": 0.0})
            old_qty = float(position["quantity"])
            old_avg = float(position["avg_entry_price"])
            new_qty = old_qty + qty
            new_avg = (old_qty * old_avg + qty * fill_price) / new_qty if new_qty > 0 else 0.0

            self._positions[symbol] = {
                "quantity": new_qty,
                "avg_entry_price": new_avg,
            }
            self._balance -= total_cost

            self._record_trade(
                side="BUY",
                symbol=symbol,
                qty=qty,
                requested_price=price,
                fill_price=fill_price,
                fee=fee,
                realized_pnl=None,
            )
            self._persist_state()

            logger.info(
                f"[PAPER BUY] {symbol} qty={qty:.8f} price={fill_price:.8f} "
                f"fee={fee:.6f} balance={self._balance:.4f}"
            )

            return {
                "status": "filled",
                "symbol": symbol,
                "side": "buy",
                "quantity": qty,
                "fill_price": fill_price,
                "fee": fee,
                "latency_ms": 0.0,
            }

    def sell(self, symbol: str, qty: float, price: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            if symbol not in self._positions or self._positions[symbol]["quantity"] <= 0:
                raise ValueError(f"No open paper position for {symbol}")

            position = self._positions[symbol]
            available_qty = float(position["quantity"])
            if qty > available_qty + 1e-12:
                raise ValueError(f"Sell quantity {qty} exceeds position {available_qty}")

            fill_price = self._market_price(symbol, "sell", price)
            fee = fill_price * qty * self.fee_rate
            proceeds = fill_price * qty - fee
            avg_entry = float(position["avg_entry_price"])
            realized_pnl = proceeds - (avg_entry * qty)

            self._balance += proceeds
            self._realized_pnl += realized_pnl

            remaining = available_qty - qty
            if remaining <= 1e-12:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol]["quantity"] = remaining

            self._record_trade(
                side="SELL",
                symbol=symbol,
                qty=qty,
                requested_price=price,
                fill_price=fill_price,
                fee=fee,
                realized_pnl=realized_pnl,
            )
            self._persist_state()

            logger.info(
                f"[PAPER SELL] {symbol} qty={qty:.8f} price={fill_price:.8f} "
                f"fee={fee:.6f} pnl={realized_pnl:+.6f} balance={self._balance:.4f}"
            )

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

    def close_position(self, symbol: str) -> Dict[str, Any]:
        position = self._positions.get(symbol)
        if not position or position["quantity"] <= 0:
            return {"status": "no_position", "symbol": symbol, "quantity": 0.0}
        return self.sell(symbol, float(position["quantity"]))

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

    def cancel_order(self, order_id: str) -> bool:
        logger.debug(f"[PAPER CANCEL] order={order_id}")
        return True

    def cancel_all_orders(self, symbol: str) -> None:
        logger.debug(f"[PAPER CANCEL_ALL] symbol={symbol}")

    def reconcile_positions(self) -> None:
        self._load_state()
        logger.info(
            f"[PAPER] Reconciled state: balance={self._balance:.4f}, positions={len(self._positions)}"
        )
