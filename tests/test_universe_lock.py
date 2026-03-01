from __future__ import annotations

from unittest.mock import patch

from config.settings import TRADING_UNIVERSE
from engine.decision import DecisionEngine
from tests.conftest import make_mock_fetcher, make_mock_predictor


def test_universe_locked_to_xrp_only() -> None:
    assert TRADING_UNIVERSE == ["XRP-USDT"]

    fetcher = make_mock_fetcher(current_price=0.6)
    predictor = make_mock_predictor(prob_up=0.7, prob_down=0.3, confidence=0.75)
    predictor.get_supported_pairs.return_value = ["XRP-USDT"]

    decision = DecisionEngine(fetcher, predictor)
    candidates = decision._get_candidate_pairs()
    assert candidates == [{"symbol": "XRP-USDT", "volValue": "0"}]

    predictor.get_supported_pairs.return_value = ["ADA-USDT"]
    with patch("engine.decision.compute_features", side_effect=lambda df, btc: df):
        signal = decision.generate_signal()
        assert signal is None
