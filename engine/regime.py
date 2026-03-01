"""
Market regime detection utilities for production decisioning.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger

try:
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator
    from ta.volatility import AverageTrueRange

    HAS_TA = True
except Exception:
    HAS_TA = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_finite(value: float) -> bool:
    return np.isfinite(value)


class MarketRegimeDetector:
    """
    Detects high-level market regimes using trend, volatility, and momentum.
    """

    def __init__(
        self,
        ema_fast_period: int = 50,
        ema_slow_period: int = 200,
        atr_period: int = 14,
        atr_mean_window: int = 100,
        rsi_period: int = 14,
        trend_threshold: float = 0.003,
        high_vol_ratio: float = 1.20,
    ) -> None:
        self.ema_fast_period = int(ema_fast_period)
        self.ema_slow_period = int(ema_slow_period)
        self.atr_period = int(atr_period)
        self.atr_mean_window = int(atr_mean_window)
        self.rsi_period = int(rsi_period)
        self.trend_threshold = float(trend_threshold)
        self.high_vol_ratio = float(high_vol_ratio)

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return market regime snapshot.
        """
        default = {
            "trend": "neutral",
            "volatility": "low",
            "momentum": "weak",
            "regime_score": 0.0,
        }

        if not isinstance(df, pd.DataFrame) or df.empty:
            return default

        required_cols = {"close", "high", "low"}
        if not required_cols.issubset(df.columns):
            return default

        try:
            working = df.copy()
            for col in required_cols:
                working[col] = pd.to_numeric(working[col], errors="coerce")
            working.dropna(subset=["close", "high", "low"], inplace=True)
            if working.empty:
                return default

            min_rows = max(self.ema_slow_period, self.atr_period, self.rsi_period) + 2
            if len(working) < min_rows:
                return default

            ema_fast = self._ema(working["close"], self.ema_fast_period)
            ema_slow = self._ema(working["close"], self.ema_slow_period)
            atr = self._atr(working["high"], working["low"], working["close"], self.atr_period)
            atr_mean = atr.rolling(
                self.atr_mean_window,
                min_periods=max(20, self.atr_mean_window // 2),
            ).mean()
            rsi = self._rsi(working["close"], self.rsi_period)

            ema_fast_last = _safe_float(ema_fast.iloc[-1], np.nan)
            ema_slow_last = _safe_float(ema_slow.iloc[-1], np.nan)
            atr_last = _safe_float(atr.iloc[-1], np.nan)
            atr_mean_last = _safe_float(atr_mean.iloc[-1], np.nan)
            rsi_last = _safe_float(rsi.iloc[-1], np.nan)

            if not (
                _is_finite(ema_fast_last)
                and _is_finite(ema_slow_last)
                and _is_finite(atr_last)
                and _is_finite(atr_mean_last)
                and _is_finite(rsi_last)
            ):
                return default

            trend = self._classify_trend(ema_fast_last, ema_slow_last)
            volatility = self._classify_volatility(atr_last, atr_mean_last)
            momentum = self._classify_momentum(rsi_last)
            regime_score = self._score(
                ema_fast_last=ema_fast_last,
                ema_slow_last=ema_slow_last,
                atr_last=atr_last,
                atr_mean_last=atr_mean_last,
                rsi_last=rsi_last,
                trend=trend,
            )

            return {
                "trend": trend,
                "volatility": volatility,
                "momentum": momentum,
                "regime_score": regime_score,
            }
        except Exception as exc:
            logger.debug(f"Regime detection fallback due to error: {exc}")
            return default

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        if HAS_TA:
            return EMAIndicator(close=series, window=period).ema_indicator()
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    def _atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        if HAS_TA:
            return AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()
        prev_close = close.shift(1)
        high_low = high - low
        high_close = (high - prev_close).abs()
        low_close = (low - prev_close).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period, min_periods=period).mean()

    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        if HAS_TA:
            return RSIIndicator(close=series, window=period).rsi()
        delta = series.diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)
        avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def _classify_trend(self, ema_fast: float, ema_slow: float) -> str:
        if abs(ema_slow) < 1e-12:
            return "neutral"
        rel_delta = (ema_fast - ema_slow) / abs(ema_slow)
        if rel_delta > self.trend_threshold:
            return "bull"
        if rel_delta < -self.trend_threshold:
            return "bear"
        return "neutral"

    def _classify_volatility(self, atr_value: float, atr_mean_value: float) -> str:
        if atr_mean_value <= 1e-12:
            return "low"
        ratio = atr_value / atr_mean_value
        return "high" if ratio >= self.high_vol_ratio else "low"

    @staticmethod
    def _classify_momentum(rsi_value: float) -> str:
        if rsi_value >= 60.0 or rsi_value <= 40.0:
            return "strong"
        return "weak"

    def _score(
        self,
        ema_fast_last: float,
        ema_slow_last: float,
        atr_last: float,
        atr_mean_last: float,
        rsi_last: float,
        trend: str,
    ) -> float:
        trend_strength = min(
            abs((ema_fast_last - ema_slow_last) / max(abs(ema_slow_last), 1e-12)) / 0.03,
            1.0,
        )
        vol_ratio = atr_last / max(atr_mean_last, 1e-12)
        vol_strength = min(abs(vol_ratio - 1.0) / 0.5, 1.0)
        momentum_strength = min(abs(rsi_last - 50.0) / 25.0, 1.0)

        alignment_bonus = 0.0
        if trend == "bull" and rsi_last >= 55.0:
            alignment_bonus = 0.10
        elif trend == "bear" and rsi_last <= 45.0:
            alignment_bonus = 0.10

        score = (
            0.45 * trend_strength
            + 0.30 * vol_strength
            + 0.25 * momentum_strength
            + alignment_bonus
        )
        return float(round(max(0.0, min(1.0, score)), 4))

