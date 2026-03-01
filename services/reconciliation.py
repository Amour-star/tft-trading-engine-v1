"""
Event-sourced reconciliation service.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

from config.settings import XRP_ONLY_SYMBOL
from data.database import Position, PositionEvent, Trade, get_session

if TYPE_CHECKING:
    from execution.base_executor import BaseExecutor


RECONCILE_REQUEST = "reconcile_request"
RECONCILE_COMPLETE = "reconcile_complete"
RECONCILE_FAILED = "reconcile_failed"
TERMINAL_EVENTS = {RECONCILE_COMPLETE, RECONCILE_FAILED}


def _structured_reconcile_log(symbol: str, event_type: str, details: Dict[str, Any]) -> None:
    logger.bind(
        reconciliation={
            "symbol": symbol,
            "event": event_type,
            "details": details,
        }
    ).info("RECONCILIATION_EVENT")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_request_processed(request_event: PositionEvent, terminal_events: List[PositionEvent]) -> bool:
    request_id = str(request_event.id)
    for event in terminal_events:
        details = event.details or {}
        if str(details.get("request_event_id", "")) == request_id:
            return True
    req_details = request_event.details or {}
    return bool(req_details.get("processed", False))


def _find_terminal_for_request(
    request_event: PositionEvent,
    terminal_events: List[PositionEvent],
) -> Optional[PositionEvent]:
    request_id = str(request_event.id)
    for event in terminal_events:
        details = event.details or {}
        if str(details.get("request_event_id", "")) == request_id:
            return event
    return None


def emit_position_event(
    event_type: str,
    *,
    symbol: str,
    details: Optional[Dict[str, Any]] = None,
) -> PositionEvent:
    if event_type not in {RECONCILE_REQUEST, RECONCILE_COMPLETE, RECONCILE_FAILED}:
        raise ValueError(f"Unsupported position event type: {event_type}")

    payload: Dict[str, Any] = dict(details or {})
    event = PositionEvent(
        symbol=symbol,
        event_type=event_type,
        details=payload,
        timestamp=datetime.utcnow(),
    )

    session = get_session()
    try:
        session.add(event)
        session.commit()
        session.refresh(event)
    finally:
        session.close()

    _structured_reconcile_log(symbol, event_type, payload)
    return event


def schedule_reconciliation(symbols: List[str], *, source: str = "manual") -> List[PositionEvent]:
    events: List[PositionEvent] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol or "").strip() or XRP_ONLY_SYMBOL
        events.append(
            emit_position_event(
                RECONCILE_REQUEST,
                symbol=symbol,
                details={
                    "source": source,
                    "processed": False,
                },
            )
        )
    return events


def _mark_request_processed(
    request_event_id: str,
    *,
    terminal_event_id: str,
    terminal_event_type: str,
) -> None:
    session = get_session()
    try:
        request_event = session.query(PositionEvent).filter(PositionEvent.id == request_event_id).first()
        if not request_event:
            return
        details = dict(request_event.details or {})
        details.update(
            {
                "processed": True,
                "processed_at": datetime.utcnow().isoformat(),
                "result_event_id": terminal_event_id,
                "result_event_type": terminal_event_type,
            }
        )
        request_event.details = details
        request_event.timestamp = request_event.timestamp or datetime.utcnow()
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Failed to mark reconcile request as processed")
    finally:
        session.close()


def fetch_unprocessed_reconcile_requests(limit: int = 50) -> List[PositionEvent]:
    session = get_session()
    try:
        requests = (
            session.query(PositionEvent)
            .filter(PositionEvent.event_type == RECONCILE_REQUEST)
            .order_by(PositionEvent.timestamp.asc(), PositionEvent.id.asc())
            .all()
        )
        terminal = (
            session.query(PositionEvent)
            .filter(PositionEvent.event_type.in_(tuple(TERMINAL_EVENTS)))
            .all()
        )
        filtered = [event for event in requests if not _is_request_processed(event, terminal)]
        return filtered[: max(1, int(limit))]
    finally:
        session.close()


def process_reconciliation_event(event: PositionEvent, executor: "BaseExecutor") -> PositionEvent:
    symbol = str(event.symbol or XRP_ONLY_SYMBOL)
    mode = str(getattr(executor, "mode", "UNKNOWN")).upper()

    session = get_session()
    try:
        request_event = session.query(PositionEvent).filter(PositionEvent.id == event.id).first()
        if not request_event:
            return emit_position_event(
                RECONCILE_FAILED,
                symbol=symbol,
                details={
                    "reason": "request_not_found",
                    "request_event_id": str(event.id),
                    "mode": mode,
                },
            )
        if request_event.event_type != RECONCILE_REQUEST:
            return request_event

        terminal_events = (
            session.query(PositionEvent)
            .filter(PositionEvent.event_type.in_(tuple(TERMINAL_EVENTS)))
            .all()
        )
        existing_terminal = _find_terminal_for_request(request_event, terminal_events)
        if existing_terminal is not None:
            return existing_terminal
    finally:
        session.close()

    # PAPER mode is intentionally non-mutating and never raises.
    if mode == "PAPER":
        failed = emit_position_event(
            RECONCILE_FAILED,
            symbol=symbol,
            details={
                "reason": "paper_mode_noop",
                "request_event_id": str(event.id),
                "mode": mode,
            },
        )
        _mark_request_processed(
            str(event.id),
            terminal_event_id=str(failed.id),
            terminal_event_type=failed.event_type,
        )
        return failed

    try:
        session = get_session()
        try:
            open_trades = (
                session.query(Trade)
                .filter(Trade.status == "open", Trade.pair == symbol)
                .order_by(Trade.entry_time.asc(), Trade.id.asc())
                .all()
            )

            available_cash = _safe_float(executor.get_balance(), 0.0)
            required_cash = 0.0
            for trade in open_trades:
                side = str(trade.side or "BUY").upper()
                if side == "SELL":
                    required_cash += max(
                        _safe_float(trade.entry_price, 0.0) * _safe_float(trade.quantity, 0.0),
                        0.0,
                    )

            if required_cash > available_cash:
                failed = emit_position_event(
                    RECONCILE_FAILED,
                    symbol=symbol,
                    details={
                        "reason": "insufficient_balance",
                        "request_event_id": str(event.id),
                        "mode": mode,
                        "required_cash": required_cash,
                        "available_cash": available_cash,
                    },
                )
                _mark_request_processed(
                    str(event.id),
                    terminal_event_id=str(failed.id),
                    terminal_event_type=failed.event_type,
                )
                return failed

            balances = executor.fetcher.get_all_balances()
            base_asset = symbol.split("-")[0] if "-" in symbol else symbol
            exchange_base_qty = _safe_float(balances.get(base_asset, 0.0), 0.0)
            now = datetime.utcnow()
            closed_trade_ids: List[str] = []
            confirmed_trade_ids: List[str] = []

            for trade in open_trades:
                trade_qty = _safe_float(trade.quantity, 0.0)
                side = str(trade.side or "BUY").upper()
                is_orphaned = False

                if side == "BUY" and exchange_base_qty < trade_qty * 0.9:
                    is_orphaned = True

                if is_orphaned:
                    trade.status = "closed"
                    trade.exit_reason = "reconciliation"
                    trade.exit_time = now
                    try:
                        ticker = executor.fetcher.get_ticker(symbol)
                        exit_price = _safe_float(ticker.get("price"), _safe_float(trade.entry_price, 0.0))
                    except Exception:
                        exit_price = _safe_float(trade.entry_price, 0.0)
                    trade.exit_price = exit_price
                    entry_price = _safe_float(trade.entry_price, 0.0)
                    side_mult = 1.0 if side == "BUY" else -1.0
                    if entry_price > 0:
                        trade.pnl = side_mult * (exit_price - entry_price) * trade_qty
                        trade.pnl_pct = side_mult * ((exit_price - entry_price) / entry_price)
                    closed_trade_ids.append(str(trade.trade_id))
                else:
                    confirmed_trade_ids.append(str(trade.trade_id))

            remaining = (
                session.query(Trade)
                .filter(Trade.status == "open", Trade.pair == symbol)
                .all()
            )
            long_qty = sum(
                _safe_float(t.quantity, 0.0)
                for t in remaining
                if str(t.side or "BUY").upper() == "BUY"
            )
            if long_qty > 0:
                weighted_notional = sum(
                    _safe_float(t.entry_price, 0.0) * _safe_float(t.quantity, 0.0)
                    for t in remaining
                    if str(t.side or "BUY").upper() == "BUY"
                )
                avg_entry_price = weighted_notional / long_qty if long_qty > 0 else 0.0
            else:
                avg_entry_price = 0.0

            position = session.query(Position).filter(Position.symbol == symbol).first()
            if position is None:
                position = Position(symbol=symbol)
                session.add(position)
            position.quantity = exchange_base_qty
            position.avg_entry_price = avg_entry_price
            position.source_mode = mode
            position.updated_at = now

            session.commit()

            complete = emit_position_event(
                RECONCILE_COMPLETE,
                symbol=symbol,
                details={
                    "request_event_id": str(event.id),
                    "mode": mode,
                    "open_trades_seen": len(open_trades),
                    "closed_trades": closed_trade_ids,
                    "confirmed_trades": confirmed_trade_ids,
                    "available_cash": available_cash,
                    "required_cash": required_cash,
                    "exchange_base_qty": exchange_base_qty,
                },
            )
            _mark_request_processed(
                str(event.id),
                terminal_event_id=str(complete.id),
                terminal_event_type=complete.event_type,
            )
            return complete
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    except Exception as exc:
        failed = emit_position_event(
            RECONCILE_FAILED,
            symbol=symbol,
            details={
                "reason": "exception",
                "error": str(exc),
                "request_event_id": str(event.id),
                "mode": mode,
            },
        )
        _mark_request_processed(
            str(event.id),
            terminal_event_id=str(failed.id),
            terminal_event_type=failed.event_type,
        )
        return failed


class ReconciliationProcessor:
    """Background processor for reconciliation events."""

    def __init__(
        self,
        executor: "BaseExecutor",
        *,
        poll_interval_seconds: float = 2.0,
        batch_size: int = 25,
    ) -> None:
        self.executor = executor
        self.poll_interval_seconds = max(0.2, float(poll_interval_seconds))
        self.batch_size = max(1, int(batch_size))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._schedule_interval_seconds = max(
            0.0,
            float(os.getenv("RECONCILIATION_SCHEDULE_SECONDS", "0") or 0.0),
        )
        self._last_schedule_monotonic = time.monotonic()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run_loop,
            name="reconciliation-loop",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.1, float(timeout_seconds)))

    def schedule_reconciliation(self, symbols: List[str], *, source: str = "manual") -> List[PositionEvent]:
        return schedule_reconciliation(symbols, source=source)

    def process_reconciliation_event(self, event: PositionEvent) -> PositionEvent:
        return process_reconciliation_event(event, self.executor)

    def run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                mode = str(getattr(self.executor, "mode", "UNKNOWN")).upper()
                now = time.monotonic()
                if (
                    mode == "LIVE"
                    and self._schedule_interval_seconds > 0
                    and (now - self._last_schedule_monotonic) >= self._schedule_interval_seconds
                ):
                    schedule_reconciliation([XRP_ONLY_SYMBOL], source="scheduled_cycle")
                    self._last_schedule_monotonic = now

                requests = fetch_unprocessed_reconcile_requests(limit=self.batch_size)
                for request_event in requests:
                    self.process_reconciliation_event(request_event)
            except Exception:
                logger.exception("Reconciliation loop iteration failed")

            self._stop_event.wait(self.poll_interval_seconds)
