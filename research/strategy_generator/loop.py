from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from config.settings import BASE_DIR, settings
from data.database import init_db
from research.strategy_generator.backtester import StrategyResearchBacktester
from research.strategy_generator.features import StrategyFeatureBuilder
from research.strategy_generator.generator import RandomStrategyGenerator
from research.strategy_generator.store import StrategyResearchStore
from research.strategy_generator.types import ResearchRunSummary, StrategyEvaluation


class StrategyResearchLoop:
    """Continuous strategy discovery, validation, ranking, and paper deployment."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        feature_builder: Optional[StrategyFeatureBuilder] = None,
        generator: Optional[RandomStrategyGenerator] = None,
        backtester: Optional[StrategyResearchBacktester] = None,
        store: Optional[StrategyResearchStore] = None,
    ) -> None:
        init_db()
        self.data_dir = data_dir or (BASE_DIR / "data" / "historical")
        self.feature_builder = feature_builder or StrategyFeatureBuilder()
        self.generator = generator or RandomStrategyGenerator()
        self.backtester = backtester or StrategyResearchBacktester(
            initial_balance=settings.trading.paper_starting_balance,
            fee_rate=settings.trading.paper_fee_rate,
            slippage_bps=settings.trading.paper_slippage_bps or settings.backtest.slippage_bps,
            min_sharpe=settings.backtest.min_sharpe,
            max_drawdown=0.20,
            min_profit_factor=1.3,
        )
        self.store = store or StrategyResearchStore()

    def load_history(self, symbol: str, timeframe: str = "15min") -> pd.DataFrame:
        path = self.data_dir / f"{symbol}_{timeframe}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Historical data not found: {path}")
        frame = pd.read_parquet(path)
        if "timestamp" not in frame.columns:
            raise RuntimeError(f"Historical data missing timestamp column: {path}")
        return frame

    def run_once(
        self,
        *,
        symbol: str,
        timeframe: str = "15min",
        candidate_count: Optional[int] = None,
        top_percentile: Optional[float] = None,
    ) -> ResearchRunSummary:
        started_at = datetime.utcnow()
        run_id = f"research-{uuid.uuid4().hex[:12]}"
        candidate_total = max(1, int(candidate_count or settings.research.candidate_count))
        top_pct = float(top_percentile or settings.research.top_percentile)
        top_pct = min(max(top_pct, 0.01), 0.50)

        self.store.start_run(
            run_id=run_id,
            symbol=symbol,
            timeframe=timeframe,
            candidate_count=candidate_total,
            notes={"paper_mode": True},
        )

        history = self.load_history(symbol, timeframe=timeframe)
        features = self.feature_builder.build(history)
        candidates = self.generator.generate(symbol=symbol, timeframe=timeframe, count=candidate_total)
        evaluations: List[StrategyEvaluation] = []

        for idx, candidate in enumerate(candidates, start=1):
            evaluation = self.backtester.evaluate_candidate(
                candidate,
                features,
                train_fraction=settings.research.train_fraction,
                test_fraction=settings.research.test_fraction,
                max_folds=settings.research.max_walk_forward_folds,
            )
            evaluations.append(evaluation)
            if idx % 100 == 0:
                logger.info(
                    "Research progress {} {}: evaluated {}/{} candidates",
                    symbol,
                    timeframe,
                    idx,
                    candidate_total,
                )

        accepted = [item for item in evaluations if item.test_metrics.passed]
        accepted.sort(key=lambda item: item.test_metrics.score, reverse=True)

        selected_count = max(1, int(len(evaluations) * top_pct))
        selected: List[StrategyEvaluation] = accepted[:selected_count]
        cutoff = max(1, len(evaluations))
        for rank, evaluation in enumerate(accepted, start=1):
            evaluation.rank_percentile = rank / cutoff
            evaluation.selected = evaluation in selected
        for evaluation in selected:
            evaluation.deployed = True

        self.store.save_evaluations(run_id, evaluations)
        deployments = self.store.deploy_top_strategies(run_id=run_id, strategies=selected)

        summary = ResearchRunSummary(
            run_id=run_id,
            symbol=symbol,
            timeframe=timeframe,
            candidate_count=len(evaluations),
            accepted_count=len(accepted),
            selected_count=len(selected),
            deployed_count=len(deployments),
            started_at=started_at,
            completed_at=datetime.utcnow(),
            notes={
                "targets": {
                    "min_sharpe": settings.backtest.min_sharpe,
                    "max_drawdown": 0.20,
                    "min_profit_factor": 1.3,
                },
                "paper_mode": True,
            },
        )
        self.store.complete_run(summary)
        logger.info(
            "Research run complete {} {} | candidates={} accepted={} selected={} deployed={}",
            symbol,
            timeframe,
            summary.candidate_count,
            summary.accepted_count,
            summary.selected_count,
            summary.deployed_count,
        )
        return summary

    def run_forever(
        self,
        *,
        symbol: str,
        timeframe: str = "15min",
        sleep_seconds: Optional[float] = None,
    ) -> None:
        delay = float(sleep_seconds or settings.research.loop_sleep_seconds)
        while True:
            try:
                self.run_once(symbol=symbol, timeframe=timeframe)
            except Exception as exc:
                logger.exception("Continuous research loop failed for {} {}: {}", symbol, timeframe, exc)
            time.sleep(max(5.0, delay))
