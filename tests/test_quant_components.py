from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from quant.config import QuantEngineConfig
from quant.execution import ExecutionEngine
from quant.features import FeatureEngineeringEngine
from quant.regime import MarketRegimeAI
from quant.strategy import StrategyEngine
from quant.types import MarketSnapshot, RegimeState, StrategySignal


def _sample_frame(rows: int = 220, base: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range(end=datetime.utcnow(), periods=rows, freq="1min")
    close = pd.Series([base + i * 0.02 for i in range(rows)])
    high = close + 0.05
    low = close - 0.05
    volume = pd.Series([100 + (i % 10) * 2 for i in range(rows)])
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_feature_regime_strategy_pipeline() -> None:
    frame_1m = _sample_frame(260, 120.0)
    frame_5m = _sample_frame(260, 120.0)
    frame_15m = _sample_frame(260, 120.0)
    snapshot = MarketSnapshot(
        symbol="BTC-USDT",
        timestamp=datetime.utcnow(),
        ticker_price=125.0,
        best_bid=124.9,
        best_ask=125.1,
        spread_pct=0.0016,
        orderbook_imbalance=0.18,
        volume_imbalance=0.09,
        funding_rate=0.0001,
        realized_volatility=0.25,
        frames={"1min": frame_1m, "5min": frame_5m, "15min": frame_15m},
    )

    features = FeatureEngineeringEngine().compute(snapshot)
    regime = MarketRegimeAI().classify(features)
    signal = StrategyEngine(QuantEngineConfig().normalized()).generate_signal(snapshot, features, regime)

    assert isinstance(features.normalized_features, dict)
    assert regime.label in {"Trending", "Mean Reverting", "High Volatility", "Low Volatility"}
    assert signal.symbol == "BTC-USDT"
    assert -1 <= signal.direction <= 1
    assert 0.0 <= signal.confidence <= 1.0


def test_execution_open_close_roundtrip(patch_db, monkeypatch) -> None:
    monkeypatch.setattr("quant.execution.get_session", patch_db)
    cfg = QuantEngineConfig().normalized()
    engine = ExecutionEngine(cfg)
    signal = StrategySignal(
        symbol="ETH-USDT",
        timestamp=datetime.utcnow(),
        direction=1,
        confidence=0.84,
        score=0.67,
        strategy_name="momentum_breakout",
        regime="Trending",
        reason="unit-test",
    )
    regime = RegimeState(
        symbol="ETH-USDT",
        timestamp=datetime.utcnow(),
        label="Trending",
        confidence=0.8,
        position_size_mult=1.1,
        threshold_shift=-0.02,
        aggressiveness_mult=1.2,
    )
    opened = engine.process_signal(
        signal=signal,
        regime=regime,
        mark_price=2000.0,
        spread_pct=0.001,
        max_notional=1200.0,
    )
    assert opened
    trade_id = opened[0].trade_id

    # Flip signal to trigger close path.
    close_signal = StrategySignal(
        symbol="ETH-USDT",
        timestamp=datetime.utcnow() + timedelta(minutes=1),
        direction=-1,
        confidence=0.92,
        score=-0.9,
        strategy_name="mean_reversion",
        regime="High Volatility",
        reason="flip",
    )
    closed = engine.process_signal(
        signal=close_signal,
        regime=regime,
        mark_price=2010.0,
        spread_pct=0.001,
        max_notional=1200.0,
    )
    assert any(report.trade_id == trade_id and report.status == "closed" for report in closed)


def test_quant_config_contains_multi_coin_defaults() -> None:
    cfg = QuantEngineConfig().normalized()
    assert len(cfg.universe) >= 7
    assert "BTC-USDT" in cfg.universe
    assert cfg.market_interval_sec == 10 or cfg.market_interval_sec > 0
