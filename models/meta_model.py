"""
XGBoost meta-model for trade selection on top of TFT forecasts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from data.database import get_latest_model_version


VOLATILITY_REGIME_MAP = {
    "low": 0.0,
    "normal": 1.0,
    "high": 2.0,
    "extreme": 3.0,
}


def _default_meta_model_path() -> Path:
    return Path(os.getenv("META_MODEL_PATH", "models/meta/latest_xgb.pkl"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class MetaModelPrediction:
    probability: float
    confidence_score: float
    model_version: str
    features: Dict[str, float]


class XGBoostMetaModel:
    """
    Meta-model that decides whether a TFT forecast should be traded.

    If model artifact is unavailable, it falls back to TFT confidence-based probability
    so the engine can continue operating safely.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else _default_meta_model_path()
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.model_version = "xgb_meta_unloaded"
        self.feature_columns: List[str] = []

    def _resolve_artifact_path(self) -> Path:
        try:
            latest = get_latest_model_version("xgb_meta")
            if latest and latest.path:
                return Path(latest.path)
        except Exception:
            pass
        return self.model_path

    def load(self) -> bool:
        artifact_path = self._resolve_artifact_path()
        if not artifact_path.exists():
            self.model = None
            self.feature_columns = self.default_feature_columns()
            self.model_version = "xgb_meta_fallback"
            return False

        artifact = joblib.load(artifact_path)
        self.model = artifact.get("model")
        self.feature_columns = artifact.get("feature_columns", self.default_feature_columns())
        self.model_version = artifact.get("model_version", artifact_path.stem)
        self.model_path = artifact_path
        return self.model is not None

    @staticmethod
    def default_feature_columns() -> List[str]:
        cols = [f"tft_h{i}_ret" for i in range(1, 13)]
        cols.extend(
            [
                "tft_mean_ret",
                "tft_std_ret",
                "tft_up_ratio",
                "att_mean",
                "att_std",
                "att_peak",
                "att_consistency",
                "rsi_14",
                "atr_14",
                "volatility_20",
                "volatility_regime",
                "btc_correlation",
                "hour",
                "day_of_week",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
            ]
        )
        return cols

    def build_features(
        self,
        tft_prediction: Dict[str, Any],
        latest_features: Dict[str, Any],
        timestamp: datetime,
    ) -> Dict[str, float]:
        forecast: List[float] = [
            _safe_float(v, 0.0)
            for v in (tft_prediction.get("forecast_vector") or [])
        ]
        padded = (forecast + [0.0] * 12)[:12]

        attention = tft_prediction.get("attention_stats") or {}
        volatility_regime = str(latest_features.get("volatility_regime", "normal")).lower()

        feature_row = {
            **{f"tft_h{i + 1}_ret": _safe_float(padded[i]) for i in range(12)},
            "tft_mean_ret": _safe_float(np.mean(padded)),
            "tft_std_ret": _safe_float(np.std(padded)),
            "tft_up_ratio": _safe_float(np.mean(np.array(padded) > 0.0)),
            "att_mean": _safe_float(attention.get("mean", 0.0)),
            "att_std": _safe_float(attention.get("std", 0.0)),
            "att_peak": _safe_float(attention.get("peak", 0.0)),
            "att_consistency": _safe_float(attention.get("consistency", 0.0)),
            "rsi_14": _safe_float(latest_features.get("rsi_14", 50.0)),
            "atr_14": _safe_float(latest_features.get("atr_14", 0.0)),
            "volatility_20": _safe_float(latest_features.get("volatility_20", 0.0)),
            "volatility_regime": _safe_float(VOLATILITY_REGIME_MAP.get(volatility_regime, 1.0), 1.0),
            "btc_correlation": _safe_float(latest_features.get("btc_correlation", 0.0)),
            "hour": float(timestamp.hour),
            "day_of_week": float(timestamp.weekday()),
            "hour_sin": _safe_float(latest_features.get("hour_sin", 0.0)),
            "hour_cos": _safe_float(latest_features.get("hour_cos", 0.0)),
            "dow_sin": _safe_float(latest_features.get("dow_sin", 0.0)),
            "dow_cos": _safe_float(latest_features.get("dow_cos", 0.0)),
        }
        return feature_row

    def _fallback_prediction(self, tft_prediction: Dict[str, Any], features: Dict[str, float]) -> MetaModelPrediction:
        base_probability = _safe_float(tft_prediction.get("confidence", 0.0), 0.0)
        confidence_score = abs(base_probability - 0.5) * 2.0
        return MetaModelPrediction(
            probability=max(0.0, min(1.0, base_probability)),
            confidence_score=max(0.0, min(1.0, confidence_score)),
            model_version=self.model_version,
            features=features,
        )

    def predict(self, tft_prediction: Dict[str, Any], features: Dict[str, float]) -> MetaModelPrediction:
        if self.model is None and not self.load():
            return self._fallback_prediction(tft_prediction, features)

        x = np.asarray(
            [[_safe_float(features.get(col, 0.0), 0.0) for col in self.feature_columns]],
            dtype=float,
        )

        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(x)[0, 1])
        else:
            probability = float(self.model.predict(x)[0])

        confidence_score = float(abs(probability - 0.5) * 2.0)
        return MetaModelPrediction(
            probability=max(0.0, min(1.0, probability)),
            confidence_score=max(0.0, min(1.0, confidence_score)),
            model_version=self.model_version,
            features=features,
        )
