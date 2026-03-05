from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans

from quant.types import FeaturePacket, RegimeState


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


class MarketRegimeAI:
    """Hybrid clustering + rule fallback market regime classifier."""

    def __init__(self) -> None:
        self._history: Dict[str, Deque[Tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=2000)
        )
        self._models: Dict[str, KMeans] = {}
        self._cluster_labels: Dict[str, Dict[int, str]] = {}

    def classify(self, packet: FeaturePacket) -> RegimeState:
        symbol = packet.symbol
        n = packet.normalized_features
        trend = _safe_float(n.get("trend_strength", 0.0))
        vol = abs(_safe_float(n.get("realized_volatility", 0.0)))
        reversion_hint = -abs(_safe_float(n.get("vwap_deviation", 0.0)))
        self._history[symbol].append((trend, vol, reversion_hint))

        label, confidence = self._cluster_predict(symbol, trend, vol, reversion_hint)
        params = self._regime_parameters(label)
        return RegimeState(
            symbol=symbol,
            timestamp=packet.timestamp,
            label=label,
            confidence=confidence,
            position_size_mult=params["position_size_mult"],
            threshold_shift=params["threshold_shift"],
            aggressiveness_mult=params["aggressiveness_mult"],
            metadata={
                "trend_score": trend,
                "volatility_score": vol,
                "reversion_score": reversion_hint,
            },
        )

    def _cluster_predict(
        self,
        symbol: str,
        trend: float,
        vol: float,
        reversion_hint: float,
    ) -> Tuple[str, float]:
        hist = list(self._history[symbol])
        if len(hist) < 80:
            return self._rule_based_regime(trend, vol, reversion_hint), 0.55

        model = self._models.get(symbol)
        data = np.asarray(hist[-600:], dtype=np.float64)
        if model is None:
            model = KMeans(n_clusters=4, n_init=10, random_state=42)
            model.fit(data)
            self._models[symbol] = model
            self._cluster_labels[symbol] = self._map_cluster_labels(model.cluster_centers_)
        else:
            model.fit(data)
            self._cluster_labels[symbol] = self._map_cluster_labels(model.cluster_centers_)

        point = np.asarray([[trend, vol, reversion_hint]], dtype=np.float64)
        cluster = int(model.predict(point)[0])
        centroid = model.cluster_centers_[cluster]
        distance = float(np.linalg.norm(point[0] - centroid))
        confidence = max(0.2, min(0.99, 1.0 / (1.0 + distance)))
        label = self._cluster_labels[symbol].get(cluster, "Low Volatility")
        return label, confidence

    def _map_cluster_labels(self, centers: np.ndarray) -> Dict[int, str]:
        labels: Dict[int, str] = {}
        if centers.size == 0:
            return labels
        vol_idx = 1
        trend_idx = 0
        reversion_idx = 2
        vols = centers[:, vol_idx]
        trends = centers[:, trend_idx]
        rev = centers[:, reversion_idx]
        high_vol_cluster = int(np.argmax(vols))
        low_vol_cluster = int(np.argmin(vols))
        trend_cluster = int(np.argmax(np.abs(trends)))
        mean_rev_cluster = int(np.argmax(rev))
        labels[high_vol_cluster] = "High Volatility"
        labels[low_vol_cluster] = "Low Volatility"
        labels[trend_cluster] = "Trending"
        labels[mean_rev_cluster] = "Mean Reverting"
        for idx in range(len(centers)):
            labels.setdefault(idx, "Low Volatility")
        return labels

    def _rule_based_regime(self, trend: float, vol: float, reversion_hint: float) -> str:
        if vol > 1.2:
            return "High Volatility"
        if abs(trend) > 0.8:
            return "Trending"
        if reversion_hint > -0.4:
            return "Mean Reverting"
        return "Low Volatility"

    def _regime_parameters(self, regime: str) -> Dict[str, float]:
        table = {
            "Trending": {
                "position_size_mult": 1.25,
                "threshold_shift": -0.03,
                "aggressiveness_mult": 1.20,
            },
            "Mean Reverting": {
                "position_size_mult": 0.95,
                "threshold_shift": 0.02,
                "aggressiveness_mult": 0.95,
            },
            "High Volatility": {
                "position_size_mult": 0.75,
                "threshold_shift": 0.06,
                "aggressiveness_mult": 0.80,
            },
            "Low Volatility": {
                "position_size_mult": 1.05,
                "threshold_shift": -0.01,
                "aggressiveness_mult": 1.00,
            },
        }
        return table.get(regime, table["Low Volatility"])
