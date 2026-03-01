"""
Sanity checks for SPOT-only decision pipeline.

Tests:
  a) All candidates bearish → returns None + persists NO_TRADE event
  b) At least one bullish candidate → returns BUY signal
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers to build mock predictions
# ---------------------------------------------------------------------------

def _bearish_prediction(**overrides):
    base = {
        "prob_up": 0.35,
        "prob_down": 0.65,
        "expected_move": -0.005,
        "confidence": 0.70,
        "meta_probability": 0.72,
        "meta_confidence": 0.70,
        "meta_model_version": "test_xgb_v1",
        "model_version": "test_tft_v1",
        "forecast_vector": [0.0] * 6,
        "spread_pct": 0.001,
        "volume_24h": 1_000_000,
    }
    base.update(overrides)
    return base


def _bullish_prediction(**overrides):
    base = {
        "prob_up": 0.72,
        "prob_down": 0.28,
        "expected_move": 0.008,
        "confidence": 0.75,
        "meta_probability": 0.78,
        "meta_confidence": 0.75,
        "meta_model_version": "test_xgb_v1",
        "model_version": "test_tft_v1",
        "forecast_vector": [0.0] * 6,
        "spread_pct": 0.001,
        "volume_24h": 1_000_000,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dependencies():
    """Patch heavy dependencies so DecisionEngine can be instantiated."""
    with (
        patch("engine.decision.KuCoinDataFetcher") as MockFetcher,
        patch("engine.decision.TFTPredictor") as MockPredictor,
        patch("engine.decision.XGBoostMetaModel") as MockMeta,
        patch("engine.decision.RegimeEngine") as MockRegime,
        patch("engine.decision.AdaptiveConfidenceThreshold") as MockThreshold,
        patch("engine.decision.get_session") as mock_get_session,
    ):
        # Fetcher mocks
        fetcher = MockFetcher.return_value
        fetcher.get_top_usdt_pairs.return_value = [
            {"symbol": "XRP-USDT", "volValue": 2_000_000},
        ]
        fetcher.get_spread.return_value = (0.001, 0.001)
        fetcher.get_ticker.return_value = {
            "price": 0.6,
            "best_ask": 0.601,
            "best_bid": 0.599,
        }

        # Build a realistic DataFrame for fetch_klines
        import numpy as np
        import pandas as pd

        n_rows = 300
        timestamps = pd.date_range(
            end=datetime.utcnow(), periods=n_rows, freq="15min"
        )
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.random.uniform(2900, 3100, n_rows),
            "high": np.random.uniform(3050, 3200, n_rows),
            "low": np.random.uniform(2800, 2950, n_rows),
            "close": np.random.uniform(2900, 3100, n_rows),
            "volume": np.random.uniform(100, 1000, n_rows),
        })
        fetcher.fetch_klines.return_value = df

        # Predictor mock — will be overridden per test
        predictor = MockPredictor.return_value
        predictor.get_supported_pairs.return_value = ["XRP-USDT"]

        # Meta model mock
        meta = MockMeta.return_value
        meta_result = MagicMock()
        meta_result.probability = 0.72
        meta_result.confidence_score = 0.70
        meta_result.model_version = "test_xgb_v1"
        meta.build_features.return_value = {}
        meta.predict.return_value = meta_result

        # Regime detector mock
        regime = MockRegime.return_value
        regime.classify.return_value = {
            "trend": "neutral",
            "volatility": "normal",
            "momentum": "weak",
            "regime_score": 0.5,
        }
        regime.scale_threshold_by_regime.return_value = 0.55
        regime.position_size_multiplier.return_value = 1.0

        # Adaptive threshold
        threshold = MockThreshold.return_value
        threshold.compute.return_value = 0.55
        threshold.last_threshold = 0.55

        # DB session mock
        mock_session = MagicMock()
        mock_session.query.return_value.order_by.return_value.first.return_value = None
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        mock_get_session.return_value = mock_session

        yield {
            "fetcher": fetcher,
            "predictor": predictor,
            "meta": meta,
            "meta_result": meta_result,
            "regime": regime,
            "threshold": threshold,
            "session": mock_session,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpotOnlyDecision:
    """Test the SPOT-only candidate iteration logic."""

    def test_all_bearish_returns_none_and_persists_event(self, mock_dependencies):
        """When all candidates are bearish, generate_signal returns None
        and a NO_TRADE DecisionEvent is persisted."""
        from engine.decision import DecisionEngine

        mocks = mock_dependencies
        # Make predictor always return bearish
        mocks["predictor"].predict.return_value = _bearish_prediction()
        # Raise meta probability above threshold so pairs aren't disqualified
        mocks["meta_result"].probability = 0.72

        with patch("engine.decision.compute_features", side_effect=lambda df, btc: df):
            engine = DecisionEngine(mocks["fetcher"], mocks["predictor"])
            engine.allow_shorts = False
            signal = engine.generate_signal()

        assert signal is None, "Expected None when all candidates are bearish"

        # Verify session.add was called with a DecisionEvent
        add_calls = mocks["session"].add.call_args_list
        decision_events = [
            call for call in add_calls
            if hasattr(call[0][0], "status") and hasattr(call[0][0], "reason")
        ]
        assert len(decision_events) >= 1, "Expected at least one DecisionEvent to be persisted"
        event = decision_events[-1][0][0]
        assert event.status == "no_trade"
        assert "no_trade" in event.reason or "disqualified" in event.reason

    def test_bullish_candidate_returns_buy_signal(self, mock_dependencies):
        """When at least one candidate is bullish, generate_signal returns a BUY signal."""
        from engine.decision import DecisionEngine

        mocks = mock_dependencies
        # Make predictor return bullish
        mocks["predictor"].predict.return_value = _bullish_prediction()
        mocks["meta_result"].probability = 0.78

        with patch("engine.decision.compute_features", side_effect=lambda df, btc: df):
            engine = DecisionEngine(mocks["fetcher"], mocks["predictor"])
            engine.allow_shorts = False
            signal = engine.generate_signal()

        assert signal is not None, "Expected a signal when bullish candidate exists"
        assert signal.side == "BUY"
        assert signal.pair == "XRP-USDT"

    def test_mixed_candidates_skips_bearish_takes_bullish(self, mock_dependencies):
        """Single-symbol universe should still produce a bullish XRP signal."""
        from engine.decision import DecisionEngine

        mocks = mock_dependencies

        call_count = {"n": 0}
        def _predict_side_effect(df, pair):
            call_count["n"] += 1
            if pair != "XRP-USDT":
                raise AssertionError(f"Unexpected pair in XRP-only mode: {pair}")
            return _bullish_prediction()

        mocks["predictor"].predict.side_effect = _predict_side_effect
        mocks["meta_result"].probability = 0.75

        with patch("engine.decision.compute_features", side_effect=lambda df, btc: df):
            engine = DecisionEngine(mocks["fetcher"], mocks["predictor"])
            engine.allow_shorts = False
            signal = engine.generate_signal()

        assert signal is not None, "Expected a BUY signal from a non-bearish candidate"
        assert signal.side == "BUY"

    def test_no_candidates_returns_none_with_event(self, mock_dependencies):
        """When no candidate pairs exist, returns None and persists event."""
        from engine.decision import DecisionEngine

        mocks = mock_dependencies
        
        with patch("engine.decision.compute_features", side_effect=lambda df, btc: df):
            engine = DecisionEngine(mocks["fetcher"], mocks["predictor"])
            with patch.object(engine, "_get_candidate_pairs", return_value=[]):
                signal = engine.generate_signal()

        assert signal is None

        add_calls = mocks["session"].add.call_args_list
        decision_events = [
            call for call in add_calls
            if hasattr(call[0][0], "reason")
        ]
        assert len(decision_events) >= 1
        event = decision_events[-1][0][0]
        assert event.reason == "no_candidate_pairs"

    def test_invalid_prediction_is_skipped_and_not_persisted(self, mock_dependencies):
        """Invalid model prediction should skip pair and avoid Prediction inserts."""
        from engine.decision import DecisionEngine

        mocks = mock_dependencies
        mocks["predictor"].predict.return_value = {
            "prob_up": 0.5,
            "prob_down": 0.5,
            "expected_move": 0.0,
            "confidence": 0.0,
            "valid": False,
        }

        with patch("engine.decision.compute_features", side_effect=lambda df, btc: df):
            engine = DecisionEngine(mocks["fetcher"], mocks["predictor"])
            signal = engine.generate_signal()

        assert signal is None
        prediction_like_adds = [
            call for call in mocks["session"].add.call_args_list
            if hasattr(call[0][0], "timeframe") and hasattr(call[0][0], "expected_move")
        ]
        assert prediction_like_adds == []

