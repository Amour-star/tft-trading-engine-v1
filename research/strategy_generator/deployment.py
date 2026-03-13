from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

import data.database as database
from research.strategy_generator.backtester import StrategyResearchBacktester
from research.strategy_generator.features import StrategyFeatureBuilder
from research.strategy_generator.types import StrategyCandidate


@dataclass
class DeploymentSignal:
    strategy_id: str
    score: float
    confidence: float
    direction: int
    reason: str


class DeployedStrategySignalProvider:
    """Loads active research deployments and evaluates them on live paper snapshots."""

    def __init__(self, refresh_seconds: float = 30.0) -> None:
        self.refresh_seconds = max(5.0, float(refresh_seconds))
        self.feature_builder = StrategyFeatureBuilder()
        self._backtester = StrategyResearchBacktester()
        self._cache: Dict[str, tuple[float, List[StrategyCandidate]]] = {}

    def active_candidates(self, symbol: str, timeframe: str = "15min") -> List[StrategyCandidate]:
        cache_key = f"{symbol}:{timeframe}"
        cached_at, cached_items = self._cache.get(cache_key, (0.0, []))
        if time.monotonic() - cached_at <= self.refresh_seconds:
            return list(cached_items)

        session = database.get_session()
        try:
            rows = (
                session.query(database.ResearchDeployment)
                .filter(database.ResearchDeployment.symbol == symbol)
                .filter(database.ResearchDeployment.timeframe == timeframe)
                .filter(database.ResearchDeployment.deployment_mode == "PAPER")
                .filter(database.ResearchDeployment.is_active.is_(True))
                .order_by(database.ResearchDeployment.score.desc(), database.ResearchDeployment.id.asc())
                .all()
            )
            candidates = [StrategyCandidate.from_payload(dict(row.definition_json or {})) for row in rows]
            self._cache[cache_key] = (time.monotonic(), candidates)
            return list(candidates)
        finally:
            session.close()

    def evaluate(self, symbol: str, frame: pd.DataFrame, timeframe: str = "15min") -> List[DeploymentSignal]:
        if frame.empty:
            return []
        candidates = self.active_candidates(symbol, timeframe=timeframe)
        if not candidates:
            return []

        features = self.feature_builder.build(frame)
        latest = features.iloc[-1]
        signals: List[DeploymentSignal] = []
        for candidate in candidates:
            direction, confidence, reason = self._backtester._signal(candidate, latest)  # noqa: SLF001
            if direction == 0:
                continue
            signed_score = confidence if direction > 0 else -confidence
            signals.append(
                DeploymentSignal(
                    strategy_id=candidate.strategy_id,
                    score=float(signed_score),
                    confidence=float(confidence),
                    direction=int(direction),
                    reason=reason,
                )
            )
        signals.sort(key=lambda item: abs(item.score), reverse=True)
        return signals

    def best_signal(self, symbol: str, frame: pd.DataFrame, timeframe: str = "15min") -> Optional[DeploymentSignal]:
        signals = self.evaluate(symbol, frame, timeframe=timeframe)
        return signals[0] if signals else None
