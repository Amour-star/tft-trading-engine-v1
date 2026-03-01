from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from data.database import PositionEvent, Trade
from services.reconciliation import (
    RECONCILE_COMPLETE,
    RECONCILE_FAILED,
    emit_position_event,
    process_reconciliation_event,
)


class _DummyLiveExecutor:
    mode = "LIVE"

    def __init__(self, *, balance: float, base_qty: float, mark_price: float) -> None:
        self._balance = float(balance)
        self.fetcher = MagicMock()
        self.fetcher.get_all_balances.return_value = {"XRP": float(base_qty)}
        self.fetcher.get_ticker.return_value = {"price": float(mark_price)}

    def get_balance(self) -> float:
        return self._balance


def test_reconciliation_event_replay_is_idempotent(patch_db) -> None:
    session = patch_db()
    trade = Trade(
        trade_id="replay_case_trade",
        pair="XRP-USDT",
        side="BUY",
        entry_time=datetime.utcnow(),
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        quantity=1.0,
        status="open",
        confidence=0.8,
    )
    session.add(trade)
    session.commit()
    session.close()

    request = emit_position_event("reconcile_request", symbol="XRP-USDT", details={"source": "test"})
    executor = _DummyLiveExecutor(balance=10_000.0, base_qty=0.0, mark_price=98.5)

    first_terminal = process_reconciliation_event(request, executor)
    second_terminal = process_reconciliation_event(request, executor)

    assert first_terminal.event_type == RECONCILE_COMPLETE
    assert second_terminal.id == first_terminal.id

    session = patch_db()
    try:
        terminals = (
            session.query(PositionEvent)
            .filter(PositionEvent.event_type.in_([RECONCILE_COMPLETE, RECONCILE_FAILED]))
            .all()
        )
        linked = [e for e in terminals if (e.details or {}).get("request_event_id") == request.id]
        assert len(linked) == 1

        updated_trade = session.query(Trade).filter(Trade.trade_id == "replay_case_trade").first()
        assert updated_trade is not None
        assert updated_trade.status == "closed"
        assert updated_trade.exit_reason == "reconciliation"
    finally:
        session.close()


def test_reconciliation_insufficient_balance_is_safe(patch_db) -> None:
    session = patch_db()
    short_trade = Trade(
        trade_id="insufficient_balance_trade",
        pair="XRP-USDT",
        side="SELL",
        entry_time=datetime.utcnow(),
        entry_price=200.0,
        stop_price=205.0,
        target_price=190.0,
        quantity=10.0,
        status="open",
        confidence=0.7,
    )
    session.add(short_trade)
    session.commit()
    session.close()

    request = emit_position_event("reconcile_request", symbol="XRP-USDT", details={"source": "test"})
    executor = _DummyLiveExecutor(balance=100.0, base_qty=0.0, mark_price=200.0)
    terminal = process_reconciliation_event(request, executor)

    assert terminal.event_type == RECONCILE_FAILED
    assert (terminal.details or {}).get("reason") == "insufficient_balance"

    session = patch_db()
    try:
        trade = session.query(Trade).filter(Trade.trade_id == "insufficient_balance_trade").first()
        assert trade is not None
        assert trade.status == "open"
    finally:
        session.close()


def test_reconcile_api_enqueues_request(patch_db) -> None:
    from fastapi.testclient import TestClient
    from dashboard.api import app

    get_session = patch_db
    api_client = TestClient(app)
    response = api_client.post("/reconcile", params={"symbol": "XRP-USDT"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["symbol"] == "XRP-USDT"
    assert payload["event_id"]

    session = get_session()
    try:
        event = session.query(PositionEvent).filter(PositionEvent.id == payload["event_id"]).first()
        assert event is not None
        assert event.event_type == "reconcile_request"
        assert event.symbol == "XRP-USDT"
    finally:
        session.close()


def test_paper_mode_reconciliation_returns_failed_without_raising(patch_db) -> None:
    _ = patch_db
    request = emit_position_event("reconcile_request", symbol="XRP-USDT", details={"source": "test"})
    paper_executor = SimpleNamespace(mode="PAPER", fetcher=MagicMock(), get_balance=lambda: 0.0)
    terminal = process_reconciliation_event(request, paper_executor)
    assert terminal.event_type == RECONCILE_FAILED
    assert (terminal.details or {}).get("reason") == "paper_mode_noop"
