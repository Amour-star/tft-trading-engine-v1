from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from data.database import DecisionEvent
from engine.decision import DecisionEngine
from engine.main import TradingEngine
from tests.conftest import make_mock_fetcher, make_mock_predictor, make_ohlcv


def test_decision_every_5min_scheduler() -> None:
    engine = TradingEngine.__new__(TradingEngine)
    engine._align_signal_to_interval = True
    engine._cycle_interval = 300
    engine._last_signal_bucket = None

    run_1, last_1 = engine._should_run_signal_cycle(
        now_dt=datetime(2026, 2, 27, 12, 0, 1, tzinfo=timezone.utc),
        now_mono=10.0,
        last_signal_time=0.0,
    )
    run_2, last_2 = engine._should_run_signal_cycle(
        now_dt=datetime(2026, 2, 27, 12, 4, 59, tzinfo=timezone.utc),
        now_mono=20.0,
        last_signal_time=last_1,
    )
    run_3, _ = engine._should_run_signal_cycle(
        now_dt=datetime(2026, 2, 27, 12, 5, 0, tzinfo=timezone.utc),
        now_mono=30.0,
        last_signal_time=last_2,
    )

    assert run_1 is True
    assert run_2 is False
    assert run_3 is True


def test_no_trade_reason_emitted(patch_db) -> None:
    engine = TradingEngine.__new__(TradingEngine)
    engine._last_cycle_reason = "startup"
    engine._no_trade_reason_counts = {}

    signal = SimpleNamespace(
        pair="XRP-USDT",
        ai_score=0.61,
        final_ai_score=0.61,
        confidence=0.61,
        prob_up=0.62,
        prob_down=0.38,
        market_regime="neutral",
        volatility_regime="normal",
        adaptive_threshold=0.57,
    )
    engine._register_no_trade_reason(
        reason_code="execution_rejected_or_failed",
        detail="max_notional_per_trade",
        signal=signal,
        persist_event=True,
    )

    session = patch_db()
    try:
        row = session.query(DecisionEvent).order_by(DecisionEvent.id.desc()).first()
        assert row is not None
        assert row.status == "no_trade"
        assert row.reason == "execution_rejected_or_failed"
        assert row.best_pair == "XRP-USDT"
    finally:
        session.close()


def test_stale_data_guard() -> None:
    fetcher = make_mock_fetcher(current_price=100.0)
    predictor = make_mock_predictor(prob_up=0.7, prob_down=0.3, confidence=0.72)

    stale_df = make_ohlcv(320, base_price=100.0, seed=7)
    stale_df["timestamp"] = pd.date_range(
        end=datetime.utcnow() - timedelta(hours=8),
        periods=len(stale_df),
        freq="15min",
    )
    fetcher.fetch_klines.return_value = stale_df

    engine = DecisionEngine(fetcher, predictor)
    evaluation = engine._evaluate_pair(
        pair="XRP-USDT",
        pair_info={"symbol": "XRP-USDT", "volValue": "1000000"},
        btc_df=stale_df,
    )

    assert evaluation.disqualified is True
    assert evaluation.disqualify_reason == "Stale market data"
