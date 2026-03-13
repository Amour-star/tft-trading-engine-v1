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
            "state": "range",
            "trend": "neutral",
            "volatility": "normal",
            "momentum": "weak",
            "regime_score": 0.0,
            "state_confidence": 0.0,
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
            atr_pct = atr / max(_safe_float(close.iloc[-1], 1.0), 1e-9)

            directional_window = close.diff().tail(20).dropna()
            directional_travel = float(directional_window.abs().sum()) if not directional_window.empty else 0.0
            net_move = abs(_safe_float(close.iloc[-1]) - _safe_float(close.iloc[-20], _safe_float(close.iloc[-1])))
            efficiency_ratio = (
                float(net_move / directional_travel)
                if directional_travel > 1e-9
                else 0.0
            )
            range_high = _safe_float(close.tail(20).max(), _safe_float(close.iloc[-1]))
            range_low = _safe_float(close.tail(20).min(), _safe_float(close.iloc[-1]))
            range_width = max(0.0, range_high - range_low)
            range_width_pct = range_width / max(_safe_float(close.iloc[-1], 1.0), 1e-9)
            trend_strength = min(
                1.0,
                max(0.0, abs(_safe_float(ema_gap.iloc[-1])) / 0.015 + abs(trend_ema_slope) / 0.0015),
            )
            chop_score = min(
                1.0,
                max(0.0, (1.0 - efficiency_ratio) * 0.65 + min(range_width_pct / max(atr_pct, 1e-6), 4.0) * 0.0875),
            )

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

            if vol_percentile >= 0.85:
                state = "high_volatility"
                state_confidence = min(1.0, 0.55 + max(0.0, vol_percentile - 0.85) * 3.0)
            elif vol_percentile <= 0.15:
                state = "low_volatility"
                state_confidence = min(1.0, 0.55 + max(0.0, 0.15 - vol_percentile) * 3.0)
            elif trend != "neutral" and trend_strength >= 0.55 and efficiency_ratio >= 0.35:
                state = "trend"
                state_confidence = min(1.0, 0.45 + trend_strength * 0.35 + efficiency_ratio * 0.20)
            elif efficiency_ratio <= 0.20 and range_width_pct <= max(atr_pct * 2.0, 0.012):
                state = "range"
                state_confidence = min(1.0, 0.50 + (0.20 - efficiency_ratio) * 1.25)
            else:
                state = "chop"
                state_confidence = min(1.0, 0.40 + chop_score * 0.45 + abs(vol_percentile - 0.5) * 0.20)

            trend_score = min(abs(trend_ema_slope) / 0.002, 1.0)
            vol_score = 1.0 - abs(vol_percentile - 0.5) * 2.0
            mom_score = min(momentum_abs / 0.0015, 1.0)
            regime_score = float(
                max(
                    0.0,
                    min(
                        1.0,
                        0.35 * trend_score
                        + 0.20 * vol_score
                        + 0.20 * mom_score
                        + 0.15 * trend_strength
                        + 0.10 * max(0.0, 1.0 - chop_score),
                    ),
                )
            )

            regime = {
                "state": state,
                "trend": trend,
                "volatility": volatility,
                "momentum": momentum,
                "regime_score": round(regime_score, 4),
                "state_confidence": round(max(0.0, min(1.0, state_confidence)), 4),
                "atr": round(max(0.0, atr), 8),
                "atr_pct": round(max(0.0, atr_pct), 8),
                "volatility_percentile": round(max(0.0, min(1.0, vol_percentile)), 4),
                "momentum_slope": round(momentum_slope, 8),
                "trend_ema_slope": round(trend_ema_slope, 8),
                "trend_strength": round(max(0.0, min(1.0, trend_strength)), 4),
                "efficiency_ratio": round(max(0.0, min(1.0, efficiency_ratio)), 4),
                "range_width_pct": round(max(0.0, range_width_pct), 8),
                "chop_score": round(max(0.0, min(1.0, chop_score)), 4),
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
        state = str(regime.get("state", "range"))
        if regime.get("volatility") == "low":
            threshold -= 0.03
        if regime.get("trend") == "bull":
            threshold -= 0.02
        if regime.get("trend") == "bear" and allow_shorts:
            threshold -= 0.02
        if regime.get("volatility") == "high":
            threshold += 0.05
        if state == "chop":
            threshold += 0.03
        elif state == "range":
            threshold += 0.01
        elif state == "trend":
            threshold -= 0.02
        return min(max(threshold, 0.40), 0.75)

    @staticmethod
    def position_size_multiplier(regime: Dict[str, Any]) -> float:
        """Regime modifies size multiplier but never reduces below 0.7 (aggressive mode)."""
        size_multiplier = 1.0
        state = str(regime.get("state", "range"))
        if regime.get("trend") == "bull":
            size_multiplier += 0.3
        elif regime.get("trend") == "bear":
            size_multiplier -= 0.2
        # Neutral trend keeps 1.0 (no penalty)
        if regime.get("volatility") == "high":
            size_multiplier -= 0.1
        if regime.get("momentum") == "strong":
            size_multiplier += 0.2
        if state == "trend":
            size_multiplier += 0.15
        elif state == "chop":
            size_multiplier -= 0.15
        elif state == "high_volatility":
            size_multiplier -= 0.20
        elif state == "low_volatility":
            size_multiplier += 0.05
        return max(0.7, min(2.0, float(size_multiplier)))
