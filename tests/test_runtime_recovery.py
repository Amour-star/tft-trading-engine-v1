from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from engine.main import TradingEngine
from services.regime_engine import RegimeEngine
from tests.conftest import make_ohlcv


def test_signal_cycle_resumes_paused_safety_after_market_recovery(monkeypatch):
    engine = TradingEngine.__new__(TradingEngine)
    engine.shutdown_controller = SimpleNamespace(should_stop=lambda: False)
    engine._last_signal_cycle_monotonic = 0.0
    engine._mark_cycle_started = MagicMock()
    engine._register_no_trade_reason = MagicMock()
    engine._record_trade_rejection = MagicMock()
    engine._mark_cycle_terminal = MagicMock()
    engine._save_engine_state = MagicMock()
    engine._refresh_market_data_health = MagicMock(
        return_value=(True, "ok", {"auth_required": False, "credentials_valid": True})
    )
    engine._accept_new_trades = False
    engine.safety = SimpleNamespace(
        load_state=MagicMock(),
        is_paused=True,
        is_killed=False,
        resume_trading=MagicMock(),
        can_trade=MagicMock(return_value=(False, "Trading is paused")),
    )
    engine.prop_risk = SimpleNamespace(estimate_equity=MagicMock(return_value=10_000.0))

    monkeypatch.setattr("engine.main.is_safe_mode", lambda: False)
    monkeypatch.setattr("engine.main.evaluate_and_arm_kill_switch", lambda equity: (False, "", {}))

    engine._signal_cycle()

    assert engine._accept_new_trades is True
    engine.safety.resume_trading.assert_called_once()


def test_regime_engine_always_returns_valid_state():
    engine = RegimeEngine()
    frame = make_ohlcv(320, base_price=100.0)
    regime = engine.classify(frame)
    assert regime["state"] in {"trend", "range", "chop", "high_volatility", "low_volatility"}
    assert regime["trend"] in {"bull", "bear", "neutral"}
    assert regime["volatility"] in {"low", "normal", "high"}
