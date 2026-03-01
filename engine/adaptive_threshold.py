"""
Adaptive confidence thresholding for live decision control.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, Iterable

from config.settings import settings


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    values_list = list(values)
    if not values_list:
        return default
    return sum(values_list) / len(values_list)


logger = logging.getLogger(__name__)


class AdaptiveConfidenceThreshold:
    """
    Computes an execution threshold based on rolling performance and regime context.
    """

    def __init__(
        self,
        base_threshold: float,
        min_threshold: float = 0.40,
        max_threshold: float = 0.75,
        window_size: int = 100,
        aggression_level: float | None = None,
    ) -> None:
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.window_size = max(10, int(window_size))
        self.base_threshold = self._clamp(_safe_float(base_threshold, 0.65))
        resolved_aggression = _safe_float(
            settings.trading.aggression_level if aggression_level is None else aggression_level,
            1.0,
        )
        self.aggression_level = max(0.1, resolved_aggression)

        self._win_rate_history: Deque[float] = deque(maxlen=self.window_size)
        self._sharpe_history: Deque[float] = deque(maxlen=self.window_size)
        self._drawdown_history: Deque[float] = deque(maxlen=self.window_size)
        self._volatility_history: Deque[str] = deque(maxlen=self.window_size)
        self.last_threshold: float = self.base_threshold

    def set_base_threshold(self, base_threshold: float) -> None:
        self.base_threshold = self._clamp(_safe_float(base_threshold, self.base_threshold))

    def compute(self, performance_metrics: Dict[str, Any], regime_info: Dict[str, Any]) -> float:
        metrics = performance_metrics or {}
        regime = regime_info or {}

        win_rate = _safe_float(metrics.get("win_rate"), -1.0)
        sharpe = _safe_float(metrics.get("sharpe"), 0.0)
        drawdown = _safe_float(metrics.get("drawdown", metrics.get("max_drawdown")), 0.0)

        if 0.0 <= win_rate <= 1.0:
            self._win_rate_history.append(win_rate)
        self._sharpe_history.append(sharpe)
        self._drawdown_history.append(max(0.0, drawdown))

        volatility = str(regime.get("volatility", "")).lower().strip()
        if volatility:
            self._volatility_history.append(volatility)

        trend = str(regime.get("trend", "")).lower().strip()
        momentum = str(regime.get("momentum", "")).lower().strip()

        threshold = self.base_threshold / self.aggression_level
        threshold = max(0.40, threshold)

        if volatility == "low":
            threshold -= 0.05

        if trend == "strong_bull":
            threshold -= 0.03
        elif trend == "strong_bear":
            threshold -= 0.03

        if momentum == "weak":
            threshold += 0.02

        self.last_threshold = self._clamp(threshold)
        logger.info(
            "Adaptive threshold computed: {threshold:.3f} (base={base:.2f} | aggression={aggr:.2f} | trend={trend} | volatility={volatility} | momentum={momentum} | win_rate={win_rate:.3f} | sharpe={sharpe:.3f})",
            threshold=self.last_threshold,
            base=self.base_threshold,
            aggr=self.aggression_level,
            trend=trend or "unknown",
            volatility=volatility or "unknown",
            momentum=momentum or "unknown",
            win_rate=_mean(self._win_rate_history, default=0.5),
            sharpe=_mean(self._sharpe_history, default=0.0),
        )
        return self.last_threshold

    def _clamp(self, value: float) -> float:
        if value < self.min_threshold:
            return self.min_threshold
        if value > self.max_threshold:
            return self.max_threshold
        return value
