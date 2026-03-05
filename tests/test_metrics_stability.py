from __future__ import annotations

from datetime import datetime, timedelta

from data.database import Trade
from engine.metrics import PerformanceTracker


def _add_closed_trade(session, trade_id: str, pnl: float, exit_offset_minutes: int) -> None:
    now = datetime.utcnow()
    exit_time = now - timedelta(minutes=exit_offset_minutes)
    entry_time = exit_time - timedelta(minutes=5)
    session.add(
        Trade(
            trade_id=trade_id,
            pair="XRP-USDT",
            side="BUY",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=1.0,
            exit_price=1.0,
            stop_price=0.9,
            target_price=1.1,
            quantity=100.0,
            pnl=pnl,
            pnl_pct=0.0,
            r_multiple=0.0,
            status="closed",
        )
    )


def test_sharpe_and_sortino_zero_on_small_sample(patch_db) -> None:
    session = patch_db()
    _add_closed_trade(session, "m_small_1", pnl=20.0, exit_offset_minutes=10)
    _add_closed_trade(session, "m_small_2", pnl=-10.0, exit_offset_minutes=5)
    session.commit()
    session.close()

    metrics = PerformanceTracker("XRP-USDT").compute_metrics()
    assert metrics["total_trades"] == 2
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["sortino_ratio"] == 0.0
    assert metrics["metrics_ready"] is False
    assert metrics["sharpe_ratio_display"] == "N/A"
    assert metrics["sortino_ratio_display"] == "N/A"
    assert metrics["win_rate_display"] == "N/A"
    assert metrics["profit_factor_display"] == "N/A"


def test_metrics_include_window_payloads_with_numeric_defaults(patch_db) -> None:
    session = patch_db()
    _add_closed_trade(session, "m_window_1", pnl=25.0, exit_offset_minutes=20)
    _add_closed_trade(session, "m_window_2", pnl=-5.0, exit_offset_minutes=90)
    _add_closed_trade(session, "m_window_3", pnl=8.0, exit_offset_minutes=23 * 60)
    session.commit()
    session.close()

    metrics = PerformanceTracker("XRP-USDT").compute_metrics()
    for key in ("lifetime", "last_1h", "last_24h"):
        assert key in metrics
        payload = metrics[key]
        assert payload["trades"] is not None
        assert payload["win_rate"] is not None
        assert payload["total_pnl"] is not None
        assert payload["average_trade"] is not None
        assert "win_rate_display" in payload
        assert "sharpe_display" in payload
        assert "sortino_display" in payload
