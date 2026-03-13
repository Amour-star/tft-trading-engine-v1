from __future__ import annotations

from datetime import datetime

import pandas as pd

from data.database import ResearchDeployment, ResearchRun, ResearchStrategy
from research.strategy_generator.deployment import DeployedStrategySignalProvider
from research.strategy_generator.features import StrategyFeatureBuilder
from research.strategy_generator.generator import RandomStrategyGenerator
from research.strategy_generator.loop import StrategyResearchLoop
from research.strategy_generator.types import PerformanceMetrics, StrategyEvaluation
from tests.conftest import make_ohlcv


def test_strategy_feature_builder_outputs_required_columns() -> None:
    builder = StrategyFeatureBuilder()
    df = make_ohlcv(320, base_price=100.0)
    features = builder.build(df)

    required = {
        "rolling_volatility",
        "vwap_deviation",
        "momentum_3",
        "momentum_5",
        "momentum_10",
        "liquidity_imbalance",
        "volume_delta",
        "orderbook_imbalance",
    }
    assert required.issubset(features.columns)
    assert not features[list(required)].isna().any().any()


def test_random_strategy_generator_defaults_to_1000_candidates() -> None:
    generator = RandomStrategyGenerator(seed=11)
    candidates = generator.generate("XRP-USDT")

    assert len(candidates) == 1000
    assert len({candidate.strategy_id for candidate in candidates}) == 1000
    assert all(candidate.indicators for candidate in candidates)


def test_strategy_research_loop_persists_results_and_deployments(patch_db, monkeypatch) -> None:
    history = make_ohlcv(260, base_price=120.0)
    loop = StrategyResearchLoop()

    monkeypatch.setattr(loop, "load_history", lambda symbol, timeframe="15min": history)

    def fake_evaluate(candidate, frame, train_fraction=0.65, test_fraction=0.20, max_folds=3):
        is_selected = candidate.strategy_id.endswith(("0000", "0001", "0002"))
        train = PerformanceMetrics(total_trades=8, sharpe_ratio=2.0, sortino_ratio=2.4, max_drawdown=0.08, win_rate=0.62, profit_factor=1.6, total_return=0.14, score=2.2, passed=True)
        test = PerformanceMetrics(
            total_trades=6,
            sharpe_ratio=1.8 if is_selected else 0.4,
            sortino_ratio=2.1 if is_selected else 0.3,
            max_drawdown=0.10 if is_selected else 0.35,
            win_rate=0.58 if is_selected else 0.40,
            profit_factor=1.5 if is_selected else 0.8,
            total_return=0.11 if is_selected else -0.03,
            score=3.0 if is_selected else -1.0,
            passed=is_selected,
            failure_reason="" if is_selected else "oos_failed",
        )
        return StrategyEvaluation(candidate=candidate, train_metrics=train, test_metrics=test)

    monkeypatch.setattr(loop.backtester, "evaluate_candidate", fake_evaluate)
    summary = loop.run_once(symbol="XRP-USDT", timeframe="15min", candidate_count=20, top_percentile=0.10)

    session = patch_db()
    try:
        assert summary.candidate_count == 20
        assert summary.accepted_count == 3
        assert summary.deployed_count == 2
        assert session.query(ResearchRun).count() == 1
        assert session.query(ResearchStrategy).count() == 20
        assert session.query(ResearchDeployment).filter(ResearchDeployment.is_active.is_(True)).count() == 2
    finally:
        session.close()


def test_deployed_strategy_signal_provider_returns_signal(patch_db) -> None:
    generator = RandomStrategyGenerator(seed=5)
    candidate = generator.generate("BTC-USDT", count=1)[0]
    candidate.entry_logic = {
        "ema": {"gap_threshold": 0.0001},
        "momentum": {"window": 3, "mode": "trend", "threshold": 0.0001},
    }
    candidate.filter_logic = {}
    candidate.min_confirmations = 1

    session = patch_db()
    try:
        session.add(
            ResearchDeployment(
                run_id="run-1",
                strategy_id=candidate.strategy_id,
                symbol="BTC-USDT",
                timeframe="15min",
                deployment_mode="PAPER",
                rank_percentile=0.01,
                score=2.5,
                is_active=True,
                deployed_at=datetime.utcnow(),
                definition_json=candidate.to_payload(),
            )
        )
        session.commit()
    finally:
        session.close()

    provider = DeployedStrategySignalProvider(refresh_seconds=0.0)
    frame = make_ohlcv(220, base_price=100.0)
    frame["close"] = pd.Series([100.0 + i * 0.15 for i in range(len(frame))])
    frame["open"] = frame["close"].shift(1).fillna(frame["close"].iloc[0])
    frame["high"] = frame["close"] + 0.05
    frame["low"] = frame["close"] - 0.05
    signal = provider.best_signal("BTC-USDT", frame, timeframe="15min")

    assert signal is not None
    assert signal.strategy_id == candidate.strategy_id
    assert signal.direction in {-1, 1}
    assert signal.confidence > 0.0
