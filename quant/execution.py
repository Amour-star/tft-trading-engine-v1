from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from data.database import EngineState, Position, Trade, get_session
from quant.config import QuantEngineConfig
from quant.types import ExecutionReport, PortfolioState, PositionSnapshot, RegimeState, StrategySignal


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


@dataclass
class TradeLot:
    trade_id: str
    symbol: str
    side: str  # BUY for long entry, SELL for short entry
    quantity: float
    entry_price: float
    stop_price: float
    target_price: float
    trailing_stop: float
    opened_at: datetime
    entry_fee: float
    strategy_name: str
    confidence: float
    metadata: Dict[str, float]

    @property
    def signed_qty(self) -> float:
        return self.quantity if self.side == "BUY" else -self.quantity


class ExecutionEngine:
    """Institutional-style paper executor with slippage, fees and partial fills."""

    def __init__(self, cfg: QuantEngineConfig) -> None:
        self.cfg = cfg
        self.initial_balance = cfg.initial_balance
        self.cash_balance = cfg.initial_balance
        self.realized_pnl = 0.0
        self.open_lots: Dict[str, TradeLot] = {}
        self._restore_state_from_db()

    def _restore_state_from_db(self) -> None:
        session = get_session()
        try:
            wallet_state = session.query(EngineState).filter(EngineState.key == "quant_wallet").first()
            if wallet_state and isinstance(wallet_state.value, dict):
                payload = wallet_state.value
                self.cash_balance = _safe_float(payload.get("cash_balance"), self.initial_balance)
                self.realized_pnl = _safe_float(payload.get("realized_pnl"), 0.0)

            rows = session.query(Trade).filter(Trade.status == "open").all()
            for row in rows:
                side = str(row.side or "BUY").upper()
                qty = abs(_safe_float(row.quantity, 0.0))
                if qty <= 0:
                    continue
                lot = TradeLot(
                    trade_id=row.trade_id,
                    symbol=str(row.pair),
                    side=side if side in {"BUY", "SELL"} else "BUY",
                    quantity=qty,
                    entry_price=_safe_float(row.entry_price, 0.0),
                    stop_price=_safe_float(row.stop_price, 0.0),
                    target_price=_safe_float(row.target_price, 0.0),
                    trailing_stop=_safe_float(row.stop_price, 0.0),
                    opened_at=row.entry_time or datetime.utcnow(),
                    entry_fee=_safe_float(row.commission, 0.0),
                    strategy_name=str((row.prediction or {}).get("strategy_name", "unknown")),
                    confidence=_safe_float(row.confidence, 0.0),
                    metadata={
                        "risk_per_trade": _safe_float((row.prediction or {}).get("risk_per_trade"), 0.0),
                    },
                )
                self.open_lots[lot.trade_id] = lot

            # If wallet state is missing, rebuild a conservative snapshot from trade history.
            if wallet_state is None:
                closed = session.query(Trade).filter(Trade.status == "closed").all()
                pnl_sum = sum(_safe_float(t.pnl, 0.0) for t in closed)
                self.realized_pnl = pnl_sum
                long_notional = sum(
                    lot.entry_price * lot.quantity + lot.entry_fee for lot in self.open_lots.values() if lot.side == "BUY"
                )
                short_notional = sum(
                    lot.entry_price * lot.quantity - lot.entry_fee for lot in self.open_lots.values() if lot.side == "SELL"
                )
                self.cash_balance = self.initial_balance + pnl_sum - long_notional + short_notional
            self._sync_positions_table(session)
            self._persist_wallet(session)
            session.commit()
        finally:
            session.close()

    def _persist_wallet(self, session) -> None:
        state = session.query(EngineState).filter(EngineState.key == "quant_wallet").first()
        payload = {
            "cash_balance": float(self.cash_balance),
            "realized_pnl": float(self.realized_pnl),
            "open_trades": int(len(self.open_lots)),
            "updated_at": datetime.utcnow().isoformat(),
        }
        if state:
            state.value = payload
            state.updated_at = datetime.utcnow()
        else:
            session.add(EngineState(key="quant_wallet", value=payload))

    def _sync_positions_table(self, session) -> None:
        by_symbol = self._aggregate_positions()
        existing = {p.symbol: p for p in session.query(Position).all()}
        for symbol, (qty, avg) in by_symbol.items():
            row = existing.get(symbol)
            if row is None:
                session.add(
                    Position(
                        symbol=symbol,
                        quantity=float(qty),
                        avg_entry_price=float(avg),
                        source_mode="PAPER",
                        updated_at=datetime.utcnow(),
                    )
                )
            else:
                row.quantity = float(qty)
                row.avg_entry_price = float(avg)
                row.updated_at = datetime.utcnow()
        for symbol, row in existing.items():
            if symbol not in by_symbol:
                session.delete(row)

    def _aggregate_positions(self) -> Dict[str, Tuple[float, float]]:
        accumulator: Dict[str, List[float]] = {}
        for lot in self.open_lots.values():
            signed_qty = lot.signed_qty
            notional = lot.entry_price * lot.quantity
            if lot.symbol not in accumulator:
                accumulator[lot.symbol] = [0.0, 0.0]
            accumulator[lot.symbol][0] += signed_qty
            accumulator[lot.symbol][1] += notional if signed_qty >= 0 else -notional

        output: Dict[str, Tuple[float, float]] = {}
        for symbol, (qty, signed_notional) in accumulator.items():
            abs_qty = abs(qty)
            avg = abs(signed_notional) / abs_qty if abs_qty > 1e-12 else 0.0
            if abs_qty > 1e-12:
                output[symbol] = (qty, avg)
        return output

    def portfolio_state(self, prices: Dict[str, float]) -> PortfolioState:
        inventory_value = 0.0
        short_liability = 0.0
        unrealized = 0.0
        positions: List[PositionSnapshot] = []
        exposure = 0.0

        for lot in self.open_lots.values():
            mark = _safe_float(prices.get(lot.symbol), lot.entry_price)
            qty = lot.quantity
            if lot.side == "BUY":
                inventory_value += mark * qty
                pnl = (mark - lot.entry_price) * qty - lot.entry_fee
            else:
                short_liability += mark * qty
                pnl = (lot.entry_price - mark) * qty - lot.entry_fee
            unrealized += pnl
            exposure += abs(mark * qty)
            positions.append(
                PositionSnapshot(
                    symbol=lot.symbol,
                    quantity=lot.signed_qty,
                    avg_entry_price=lot.entry_price,
                    stop_price=lot.stop_price,
                    take_profit=lot.target_price,
                    trailing_stop=lot.trailing_stop,
                    opened_at=lot.opened_at,
                    metadata={"trade_id": lot.trade_id, "strategy": lot.strategy_name},
                )
            )
        equity = self.cash_balance + inventory_value - short_liability
        return PortfolioState(
            timestamp=datetime.utcnow(),
            balance=float(self.cash_balance),
            realized_pnl=float(self.realized_pnl),
            unrealized_pnl=float(unrealized),
            equity=float(equity),
            open_positions=positions,
            exposure_notional=float(exposure),
        )

    def close_by_risk(
        self,
        symbol: str,
        price: float,
        reason: str,
    ) -> List[ExecutionReport]:
        reports: List[ExecutionReport] = []
        for trade_id in [tid for tid, lot in self.open_lots.items() if lot.symbol == symbol]:
            report = self._close_lot(trade_id, price=price, reason=reason)
            if report:
                reports.append(report)
        return reports

    def process_signal(
        self,
        signal: StrategySignal,
        regime: RegimeState,
        mark_price: float,
        spread_pct: float,
        max_notional: float,
    ) -> List[ExecutionReport]:
        reports: List[ExecutionReport] = []
        symbol_lots = [lot for lot in self.open_lots.values() if lot.symbol == signal.symbol]

        for lot in list(symbol_lots):
            triggered, reason = self._should_exit(lot, signal, mark_price)
            if triggered:
                closed = self._close_lot(lot.trade_id, price=mark_price, reason=reason)
                if closed:
                    reports.append(closed)

        desired_side = "BUY" if signal.direction > 0 else ("SELL" if signal.direction < 0 else "HOLD")
        if desired_side == "HOLD":
            return reports
        if desired_side == "SELL" and not self.cfg.enable_shorts and not symbol_lots:
            return reports

        same_direction_lots = [lot for lot in self.open_lots.values() if lot.symbol == signal.symbol and lot.side == desired_side]
        base_notional = max_notional * max(0.0, min(1.0, signal.confidence))
        size_adjust = max(0.2, min(2.0, regime.position_size_mult * (1.0 + abs(signal.score))))
        requested_notional = base_notional * size_adjust

        if same_direction_lots and signal.confidence < 0.67:
            return reports

        if requested_notional <= 1.0:
            return reports

        opened = self._open_lot(
            symbol=signal.symbol,
            side=desired_side,
            mark_price=mark_price,
            spread_pct=spread_pct,
            requested_notional=requested_notional,
            strategy_name=signal.strategy_name,
            confidence=signal.confidence,
            regime=regime.label,
        )
        if opened:
            reports.append(opened)
        return reports

    def _open_lot(
        self,
        symbol: str,
        side: str,
        mark_price: float,
        spread_pct: float,
        requested_notional: float,
        strategy_name: str,
        confidence: float,
        regime: str,
    ) -> Optional[ExecutionReport]:
        fill_ratio = random.uniform(self.cfg.partial_fill_min, self.cfg.partial_fill_max)
        filled_notional = requested_notional * fill_ratio
        if filled_notional <= 1.0 or mark_price <= 0:
            return None
        slip_bps = self.cfg.slippage_bps + spread_pct * 10_000.0 * random.uniform(0.8, 1.5)
        fill_price = mark_price * (1.0 + slip_bps / 10_000.0) if side == "BUY" else mark_price * (1.0 - slip_bps / 10_000.0)
        quantity = filled_notional / max(fill_price, 1e-9)
        fee = filled_notional * self.cfg.fee_rate

        if side == "BUY" and self.cash_balance < (filled_notional + fee):
            return ExecutionReport(
                trade_id="",
                symbol=symbol,
                side=side,
                status="rejected",
                filled_qty=0.0,
                avg_price=fill_price,
                fee_paid=0.0,
                slippage_bps=slip_bps,
                realized_pnl=0.0,
                message="insufficient_cash",
            )

        if side == "BUY":
            self.cash_balance -= (filled_notional + fee)
            stop = fill_price * (1.0 - 0.004)
            target = fill_price * (1.0 + 0.006)
            trailing = fill_price * (1.0 - 0.003)
        else:
            self.cash_balance += (filled_notional - fee)
            stop = fill_price * (1.0 + 0.004)
            target = fill_price * (1.0 - 0.006)
            trailing = fill_price * (1.0 + 0.003)

        trade_id = f"quant-{uuid.uuid4().hex[:16]}"
        lot = TradeLot(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=fill_price,
            stop_price=stop,
            target_price=target,
            trailing_stop=trailing,
            opened_at=datetime.utcnow(),
            entry_fee=fee,
            strategy_name=strategy_name,
            confidence=confidence,
            metadata={"regime": regime},
        )
        self.open_lots[trade_id] = lot
        self._insert_open_trade(lot, slippage_bps=slip_bps)

        logger.bind(
            event="TRADE_OPENED",
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            qty=round(quantity, 8),
            price=round(fill_price, 8),
            fee=round(fee, 8),
        ).info("TRADE_OPENED")

        return ExecutionReport(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            status="opened",
            filled_qty=quantity,
            avg_price=fill_price,
            fee_paid=fee,
            slippage_bps=slip_bps,
            realized_pnl=0.0,
            message="opened",
        )

    def _close_lot(self, trade_id: str, price: float, reason: str) -> Optional[ExecutionReport]:
        lot = self.open_lots.get(trade_id)
        if lot is None:
            return None
        slip_bps = self.cfg.slippage_bps * random.uniform(0.7, 1.3)
        fill_price = price * (1.0 - slip_bps / 10_000.0) if lot.side == "BUY" else price * (1.0 + slip_bps / 10_000.0)
        fill_notional = lot.quantity * fill_price
        fee = fill_notional * self.cfg.fee_rate

        if lot.side == "BUY":
            self.cash_balance += fill_notional - fee
            realized = (fill_price - lot.entry_price) * lot.quantity - fee - lot.entry_fee
            close_side = "SELL"
        else:
            self.cash_balance -= fill_notional + fee
            realized = (lot.entry_price - fill_price) * lot.quantity - fee - lot.entry_fee
            close_side = "BUY"

        self.realized_pnl += realized
        self.open_lots.pop(trade_id, None)
        self._update_closed_trade(
            trade_id=trade_id,
            exit_price=fill_price,
            pnl=realized,
            fee=fee,
            slippage_bps=slip_bps,
            reason=reason,
        )

        logger.bind(
            event="TRADE_CLOSED",
            trade_id=trade_id,
            symbol=lot.symbol,
            side=close_side,
            pnl=round(realized, 8),
            reason=reason,
        ).info("TRADE_CLOSED")

        return ExecutionReport(
            trade_id=trade_id,
            symbol=lot.symbol,
            side=close_side,
            status="closed",
            filled_qty=lot.quantity,
            avg_price=fill_price,
            fee_paid=fee,
            slippage_bps=slip_bps,
            realized_pnl=realized,
            message=reason,
        )

    def _should_exit(self, lot: TradeLot, signal: StrategySignal, mark_price: float) -> Tuple[bool, str]:
        if lot.side == "BUY":
            lot.trailing_stop = max(lot.trailing_stop, mark_price * (1.0 - 0.0025))
            if mark_price <= min(lot.stop_price, lot.trailing_stop):
                return True, "stop_loss"
            if mark_price >= lot.target_price:
                return True, "take_profit"
            if signal.direction < 0 and signal.confidence > 0.6:
                return True, "signal_flip"
        else:
            lot.trailing_stop = min(lot.trailing_stop, mark_price * (1.0 + 0.0025))
            if mark_price >= max(lot.stop_price, lot.trailing_stop):
                return True, "stop_loss"
            if mark_price <= lot.target_price:
                return True, "take_profit"
            if signal.direction > 0 and signal.confidence > 0.6:
                return True, "signal_flip"
        return False, ""

    def _insert_open_trade(self, lot: TradeLot, slippage_bps: float) -> None:
        session = get_session()
        try:
            trade = Trade(
                trade_id=lot.trade_id,
                pair=lot.symbol,
                side=lot.side,
                entry_time=lot.opened_at,
                entry_price=lot.entry_price,
                stop_price=lot.stop_price,
                target_price=lot.target_price,
                quantity=lot.quantity,
                status="open",
                slippage_bps=slippage_bps,
                commission=lot.entry_fee,
                confidence=lot.confidence,
                prediction={
                    "strategy_name": lot.strategy_name,
                    "risk_per_trade": lot.metadata.get("risk_per_trade", 0.0),
                    "regime": lot.metadata.get("regime", "unknown"),
                },
                ai_reasoning=f"strategy={lot.strategy_name}",
            )
            session.add(trade)
            self._sync_positions_table(session)
            self._persist_wallet(session)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.exception("Failed to persist open trade {}: {}", lot.trade_id, exc)
        finally:
            session.close()

    def _update_closed_trade(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        fee: float,
        slippage_bps: float,
        reason: str,
    ) -> None:
        session = get_session()
        try:
            row = session.query(Trade).filter(Trade.trade_id == trade_id).first()
            if row is None:
                return
            row.exit_time = datetime.utcnow()
            row.exit_price = float(exit_price)
            row.pnl = float(pnl)
            row.pnl_pct = float(pnl / max(abs(row.entry_price * row.quantity), 1e-9))
            risk = abs((row.entry_price - row.stop_price) * row.quantity)
            row.r_multiple = float(pnl / max(risk, 1e-9))
            row.exit_reason = reason
            row.status = "closed"
            row.slippage_bps = _safe_float(row.slippage_bps, 0.0) + float(slippage_bps)
            row.commission = _safe_float(row.commission, 0.0) + float(fee)
            self._sync_positions_table(session)
            self._persist_wallet(session)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.exception("Failed to persist closed trade {}: {}", trade_id, exc)
        finally:
            session.close()
