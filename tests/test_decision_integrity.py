from __future__ import annotations

import os
import tempfile
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from data.database import Trade
from engine.decision import DecisionEngine, PairEvaluation
from engine.main import TradingEngine
from execution.monitor import PositionMonitor
from execution.paper_executor import PaperExecutor
from models.meta_model import XGBoostMetaModel
from tests.conftest import make_mock_fetcher, make_mock_predictor


def test_meta_fallback_uses_directional_probability() -> None:
    model = XGBoostMetaModel(model_path="tests/fixtures/does_not_exist.pkl")

    prediction = model.predict(
        {
            "prob_up": 0.70,
            "prob_down": 0.30,
            "confidence": 0.95,
        },
        {},
    )

    assert prediction.model_version == "xgb_meta_fallback"
    assert prediction.probability == pytest.approx(0.70)
    assert prediction.confidence_score == pytest.approx(0.40)


def test_build_signal_side_follows_probabilities_not_signal_score() -> None:
    fetcher = make_mock_fetcher(current_price=100.0)
    predictor = make_mock_predictor(prob_up=0.7, prob_down=0.3, confidence=0.75)
    engine = DecisionEngine(fetcher, predictor)

    evaluation = PairEvaluation(
        pair="XRP-USDT",
        score=0.8,
        prediction={
            "prob_up": 0.70,
            "prob_down": 0.30,
            "signal_score": 0.20,
            "expected_move": 0.02,
            "confidence": 0.75,
            "meta_probability": 0.70,
            "meta_confidence": 0.40,
            "adaptive_threshold": 0.55,
            "spread_pct": 0.0002,
            "volume_24h": 1_000_000.0,
            "regime_info": {"trend": "bull", "volatility": "normal", "regime_score": 0.5},
            "regime_size_multiplier": 1.0,
            "direction_source": "probabilities",
            "expected_edge_pct": 0.0178,
            "estimated_fee_drag_pct": 0.0022,
            "directional_edge": 0.40,
            "min_risk_reward": 1.40,
            "model_version": "test_v1",
            "meta_model_version": "xgb_meta_fallback",
            "forecast_vector": [0.01] * 12,
            "allow_reasons": ["test"],
            "block_reasons": [],
            "signal_timestamp": datetime.utcnow(),
        },
        features={
            "model_close_price": 100.0,
            "atr_14": 0.5,
            "volatility_regime": "normal",
            "market_regime": "bull",
            "btc_correlation": 0.2,
            "regime_score": 0.5,
            "bb_position": 0.5,
            "momentum_10": 0.01,
        },
        reasons=["test"],
    )

    signal = engine._build_signal(evaluation)
    assert signal is not None
    assert signal.side == "BUY"
    assert signal.direction_source == "probabilities"


def test_project_target_distance_adds_structural_floor_for_paper_probe(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_ENABLE_PROBE_SIGNALS", "true")
    fetcher = make_mock_fetcher(current_price=100.0)
    predictor = make_mock_predictor(prob_up=0.69, prob_down=0.31, confidence=0.81)
    engine = DecisionEngine(fetcher, predictor)

    projected_move_pct, target_distance, target_source, structural_rr = engine._project_target_distance(
        price=100.0,
        atr=1.0,
        prediction={
            "prob_up": 0.69,
            "prob_down": 0.31,
            "expected_move": 0.002,
            "confidence": 0.81,
            "signal_score": 0.54,
            "adaptive_threshold": 0.55,
            "regime_info": {"regime_score": 0.42},
            "size_scale": 1.0,
        },
        features={"regime_score": 0.42},
    )

    assert target_source == "atr_structural_floor"
    assert structural_rr is not None
    assert structural_rr >= 1.2
    assert target_distance > 0.2
    assert projected_move_pct == pytest.approx(target_distance / 100.0)


def test_build_signal_uses_projected_target_distance_when_model_move_is_small() -> None:
    fetcher = make_mock_fetcher(current_price=100.0)
    predictor = make_mock_predictor(prob_up=0.68, prob_down=0.32, confidence=0.80)
    engine = DecisionEngine(fetcher, predictor)

    evaluation = PairEvaluation(
        pair="XRP-USDT",
        score=0.82,
        prediction={
            "prob_up": 0.68,
            "prob_down": 0.32,
            "signal_score": 0.52,
            "expected_move": 0.002,
            "confidence": 0.80,
            "meta_probability": 0.74,
            "meta_confidence": 0.45,
            "adaptive_threshold": 0.55,
            "spread_pct": 0.0002,
            "volume_24h": 1_000_000.0,
            "regime_info": {"trend": "bull", "volatility": "normal", "regime_score": 0.5},
            "regime_size_multiplier": 1.0,
            "direction_source": "probabilities",
            "projected_target_distance": 3.0,
            "projected_target_source": "atr_structural_floor",
            "projected_move_pct": 0.03,
            "projected_structural_rr": 1.5,
            "expected_edge_pct": 0.0278,
            "estimated_fee_drag_pct": 0.0022,
            "directional_edge": 0.36,
            "min_risk_reward": 1.40,
            "model_version": "test_v1",
            "meta_model_version": "xgb_meta_fallback",
            "forecast_vector": [0.01] * 12,
            "allow_reasons": ["test"],
            "block_reasons": [],
            "signal_timestamp": datetime.utcnow(),
        },
        features={
            "model_close_price": 100.0,
            "atr_14": 1.0,
            "volatility_regime": "normal",
            "market_regime": "bull",
            "btc_correlation": 0.2,
            "regime_score": 0.5,
            "bb_position": 0.5,
            "momentum_10": 0.01,
        },
        reasons=["test"],
    )

    signal = engine._build_signal(evaluation)
    assert signal is not None
    assert signal.target_price == pytest.approx(103.0)
    assert signal.risk_reward == pytest.approx(1.5)


def test_monitor_cycle_handles_multiple_open_trades(patch_db) -> None:
    session = patch_db()
    for idx, pair in enumerate(["XRP-USDT", "ETH-USDT"], start=1):
        session.add(
            Trade(
                trade_id=f"multi_trade_{idx}",
                pair=pair,
                side="BUY",
                entry_time=datetime.utcnow(),
                entry_price=100.0,
                stop_price=98.0,
                target_price=104.0,
                quantity=1.0,
                status="open",
                confidence=0.75,
            )
        )
    session.commit()
    session.close()

    fetcher = make_mock_fetcher(current_price=105.0)
    predictor = make_mock_predictor()
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    try:
        executor = PaperExecutor(fetcher=fetcher, starting_balance=10_000.0, db_path=tmp.name)
        executor.buy("XRP-USDT", qty=1.0, price=100.0)
        executor.buy("ETH-USDT", qty=1.0, price=100.0)

        monitor = PositionMonitor(fetcher, executor, predictor)
        closed_ids = monitor.monitor_cycle()

        assert isinstance(closed_ids, list)
        assert set(closed_ids) == {"multi_trade_1", "multi_trade_2"}

        session = patch_db()
        try:
            closed_count = session.query(Trade).filter(Trade.status == "closed").count()
            assert closed_count == 2
        finally:
            session.close()
    finally:
        os.unlink(tmp.name)


def test_signal_cycle_blocks_when_tft_model_unavailable() -> None:
    engine = TradingEngine.__new__(TradingEngine)
    engine.predictor = SimpleNamespace(model=None, model_version="tft_disabled")
    engine.shutdown_controller = SimpleNamespace(should_stop=lambda: False)
    engine._accept_new_trades = True
    engine._engine_state = "DEGRADED_FALLBACK"
    engine._mark_cycle_started = MagicMock()
    engine._refresh_market_data_health = MagicMock(
        return_value=(
            True,
            "",
            {
                "market_data_source": "market_data_service",
                "ticker_source": "market_data_service",
                "orderbook_source": "market_data_service",
                "auth_required": False,
                "credentials_valid": True,
            },
        )
    )
    engine.safety = SimpleNamespace(can_trade=lambda: (True, ""))
    engine.prop_risk = SimpleNamespace(estimate_equity=lambda: 10_000.0)
    engine._register_no_trade_reason = MagicMock()
    engine._record_trade_rejection = MagicMock()
    engine._mark_cycle_terminal = MagicMock()
    engine._inference_scope = lambda: nullcontext()
    engine.decision = SimpleNamespace(generate_signal=MagicMock())

    with patch("engine.main.is_safe_mode", return_value=False), patch(
        "engine.main.evaluate_and_arm_kill_switch",
        return_value=(False, "", {}),
    ), patch("engine.main.publish_event"):
        engine._signal_cycle()

    engine.decision.generate_signal.assert_not_called()
    engine._register_no_trade_reason.assert_called_once_with(
        "model_not_ready_for_live_trade",
        detail="tft_model_unavailable:tft_disabled",
    )
    engine._record_trade_rejection.assert_called_once_with("model_not_ready_for_live_trade")


def test_recover_paper_runtime_state_auto_resumes_stale_pause(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_AUTO_RESUME_ON_STARTUP", "true")

    engine = TradingEngine.__new__(TradingEngine)
    engine.mode = "PAPER"
    engine.safety = MagicMock()
    engine.safety.load_state = MagicMock()
    engine.safety.get_status.return_value = {
        "paused": True,
        "killed": False,
        "safe_mode": False,
    }
    engine.safety.resume_trading = MagicMock()
    engine._save_engine_state = MagicMock()

    TradingEngine._recover_paper_runtime_state(engine)

    engine.safety.resume_trading.assert_called_once()
    engine._save_engine_state.assert_called_once_with("last_cycle_reason", "paper_auto_resume_startup")
