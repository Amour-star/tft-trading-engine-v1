from __future__ import annotations

import random
from typing import List

from research.strategy_generator.types import StrategyCandidate


class RandomStrategyGenerator:
    """Randomly composes multi-indicator strategies for research backtests."""

    INDICATORS = (
        "rsi",
        "ema",
        "vwap",
        "atr",
        "momentum",
        "volume_spike",
        "volatility",
        "orderbook_imbalance",
    )

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)

    def generate(
        self,
        symbol: str,
        timeframe: str = "15min",
        count: int = 1000,
    ) -> List[StrategyCandidate]:
        candidates: List[StrategyCandidate] = []
        for index in range(max(1, int(count))):
            selected = self._rng.sample(
                list(self.INDICATORS),
                k=self._rng.randint(3, min(6, len(self.INDICATORS))),
            )
            entry_logic = {}
            filter_logic = {}

            for indicator in selected:
                if indicator == "rsi":
                    entry_logic[indicator] = {
                        "long_below": self._rng.randint(22, 42),
                        "short_above": self._rng.randint(58, 82),
                    }
                elif indicator == "ema":
                    entry_logic[indicator] = {
                        "gap_threshold": round(self._rng.uniform(0.0008, 0.0120), 6),
                    }
                elif indicator == "vwap":
                    entry_logic[indicator] = {
                        "mode": self._rng.choice(["trend", "mean_reversion"]),
                        "threshold": round(self._rng.uniform(0.0010, 0.0180), 6),
                    }
                elif indicator == "momentum":
                    entry_logic[indicator] = {
                        "window": self._rng.choice([3, 5, 10]),
                        "mode": self._rng.choice(["trend", "reversal"]),
                        "threshold": round(self._rng.uniform(0.0010, 0.0200), 6),
                    }
                elif indicator == "orderbook_imbalance":
                    entry_logic[indicator] = {
                        "threshold": round(self._rng.uniform(0.05, 0.45), 6),
                    }
                elif indicator == "volume_spike":
                    filter_logic[indicator] = {
                        "minimum": round(self._rng.uniform(1.05, 3.50), 6),
                    }
                elif indicator == "volatility":
                    filter_logic[indicator] = {
                        "mode": self._rng.choice(["breakout", "quiet"]),
                        "threshold": round(self._rng.uniform(0.0020, 0.0500), 6),
                    }
                elif indicator == "atr":
                    lower = round(self._rng.uniform(0.0010, 0.0200), 6)
                    upper = round(self._rng.uniform(max(lower + 0.0010, 0.0040), 0.0600), 6)
                    filter_logic[indicator] = {
                        "min_atr_pct": lower,
                        "max_atr_pct": max(lower, upper),
                    }

            min_confirmations = min(
                max(1, self._rng.randint(1, 3)),
                len(entry_logic) if entry_logic else 1,
            )
            candidates.append(
                StrategyCandidate(
                    strategy_id=f"{symbol.lower()}-{timeframe}-rand-{index:04d}",
                    symbol=symbol,
                    timeframe=timeframe,
                    indicators=selected,
                    entry_logic=entry_logic,
                    filter_logic=filter_logic,
                    min_confirmations=min_confirmations,
                    allow_short=self._rng.random() > 0.25,
                    max_hold_bars=self._rng.randint(4, 32),
                    stop_atr_multiplier=round(self._rng.uniform(0.8, 3.0), 4),
                    take_atr_multiplier=round(self._rng.uniform(1.2, 5.0), 4),
                    trailing_atr_multiplier=round(self._rng.uniform(0.6, 2.4), 4),
                    risk_per_trade=round(self._rng.uniform(0.0025, 0.0200), 5),
                    metadata={"generator": "random_indicator_mix"},
                )
            )
        return candidates
