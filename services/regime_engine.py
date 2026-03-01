"""
Regime-aware classification and scaling helpers.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _slope(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3:
        return 0.0
    x = np.arange(len(clean), dtype=float)
    y = clean.to_numpy(dtype=float)
    try:
        m, _b = np.polyfit(x, y, 1)
        return float(m)
    except Exception:
        return 0.0


class RegimeEngine:
    """
    Production-safe regime classifier plus threshold/size scaling.
    """

    def classify(self, df: pd.DataFrame) -> Dict[str, Any]:
        default = {
            "trend": "neutral",
            "volatility": "normal",
            "momentum": "weak",
            "regime_score": 0.0,
        }
        if not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
            return default

        try:
            working = df.copy()
            working["close"] = pd.to_numeric(working["close"], errors="coerce")
            working.dropna(subset=["close"], inplace=True)
            if len(working) < 30:
                return default

            close = working["close"]
            ret = close.pct_change()

            atr_raw = working.get("atr_14")
            atr_series = pd.to_numeric(atr_raw, errors="coerce") if atr_raw is not None else None
            if not isinstance(atr_series, pd.Series) or atr_series.dropna().empty:
                high = pd.to_numeric(working.get("high"), errors="coerce")
                low = pd.to_numeric(working.get("low"), errors="coerce")
                prev_close = close.shift(1)
                tr = pd.concat(
                    [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
                    axis=1,
                ).max(axis=1)
                atr_series = tr.rolling(14, min_periods=14).mean()
            atr = _safe_float(atr_series.iloc[-1], 0.0)

            vol_raw = working.get("volatility_20")
            vol_series = pd.to_numeric(vol_raw, errors="coerce") if vol_raw is not None else None
            if not isinstance(vol_series, pd.Series) or vol_series.dropna().empty:
                vol_series = ret.rolling(20, min_periods=20).std()
            vol_series = vol_series.dropna().tail(300)
            vol_now = _safe_float(vol_series.iloc[-1], 0.0) if not vol_series.empty else 0.0
            if vol_series.empty or not np.isfinite(vol_now):
                vol_percentile = 0.5
            else:
                vol_percentile = float((vol_series <= vol_now).mean())

            ema_fast = close.ewm(span=20, adjust=False).mean()
            ema_slow = close.ewm(span=50, adjust=False).mean()
            ema_gap = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
            trend_ema_slope = _slope(ema_gap.tail(20))
            momentum_slope = _slope(ret.tail(20))

            if trend_ema_slope > 0.0005:
                trend = "bull"
            elif trend_ema_slope < -0.0005:
                trend = "bear"
            else:
                trend = "neutral"

            if vol_percentile <= 0.25:
                volatility = "low"
            elif vol_percentile >= 0.75:
                volatility = "high"
            else:
                volatility = "normal"

            momentum_abs = abs(momentum_slope)
            if momentum_abs >= 0.0008:
                momentum = "strong"
            elif momentum_abs >= 0.0003:
                momentum = "moderate"
            else:
                momentum = "weak"

            trend_score = min(abs(trend_ema_slope) / 0.002, 1.0)
            vol_score = 1.0 - abs(vol_percentile - 0.5) * 2.0
            mom_score = min(momentum_abs / 0.0015, 1.0)
            regime_score = float(max(0.0, min(1.0, 0.45 * trend_score + 0.25 * vol_score + 0.30 * mom_score)))

            regime = {
                "trend": trend,
                "volatility": volatility,
                "momentum": momentum,
                "regime_score": round(regime_score, 4),
                "atr": round(max(0.0, atr), 8),
                "volatility_percentile": round(max(0.0, min(1.0, vol_percentile)), 4),
                "momentum_slope": round(momentum_slope, 8),
                "trend_ema_slope": round(trend_ema_slope, 8),
            }
            logger.bind(event="REGIME_CLASSIFIED", regime=regime).info("REGIME_CLASSIFIED")
            return regime
        except Exception as exc:
            logger.warning(f"Regime classification failed: {exc}")
            return default

    @staticmethod
    def scale_threshold_by_regime(
        base_threshold: float,
        regime: Dict[str, Any],
        allow_shorts: bool,
    ) -> float:
        threshold = float(base_threshold)
        if regime.get("volatility") == "low":
            threshold -= 0.03
        if regime.get("trend") == "bull":
            threshold -= 0.02
        if regime.get("trend") == "bear" and allow_shorts:
            threshold -= 0.02
        if regime.get("volatility") == "high":
            threshold += 0.05
        return min(max(threshold, 0.40), 0.75)

    @staticmethod
    def position_size_multiplier(regime: Dict[str, Any]) -> float:
        size_multiplier = 1.0
        if regime.get("trend") == "bull":
            size_multiplier += 0.2
        if regime.get("volatility") == "high":
            size_multiplier -= 0.3
        if regime.get("momentum") == "strong":
            size_multiplier += 0.2
        return max(0.25, min(2.0, float(size_multiplier)))
