"""
Performance metric utilities for model monitoring and retraining jobs.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _to_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.asarray([0.0], dtype=float)
    return arr


def sharpe_ratio(returns: Iterable[float], risk_free_rate: float = 0.0) -> float:
    arr = _to_array(returns)
    excess = arr - risk_free_rate
    std = float(np.std(excess, ddof=1)) if arr.size > 1 else 0.0
    if math.isclose(std, 0.0):
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(252))


def sortino_ratio(returns: Iterable[float], risk_free_rate: float = 0.0) -> float:
    arr = _to_array(returns)
    excess = arr - risk_free_rate
    downside = excess[excess < 0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    if math.isclose(downside_std, 0.0):
        return 0.0
    return float(np.mean(excess) / downside_std * math.sqrt(252))


def max_drawdown(equity_curve: Iterable[float]) -> float:
    arr = _to_array(equity_curve)
    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / np.where(peaks == 0, 1.0, peaks)
    return float(np.min(drawdowns))


def win_rate(pnls: Iterable[float]) -> float:
    arr = _to_array(pnls)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr > 0))


def profit_factor(pnls: Iterable[float]) -> float:
    arr = _to_array(pnls)
    gross_profit = float(np.sum(arr[arr > 0]))
    gross_loss = float(np.abs(np.sum(arr[arr < 0])))
    if math.isclose(gross_loss, 0.0):
        return 0.0 if math.isclose(gross_profit, 0.0) else float("inf")
    return gross_profit / gross_loss

