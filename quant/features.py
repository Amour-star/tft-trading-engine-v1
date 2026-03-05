from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from quant.types import FeaturePacket, MarketSnapshot


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(period).mean()
    losses = -delta.clip(upper=0.0).rolling(period).mean()
    rs = gains / losses.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


class FeatureEngineeringEngine:
    """Creates normalized quantitative factors for every symbol pipeline."""

    def __init__(self) -> None:
        self._scalers: Dict[str, StandardScaler] = {}
        self._history: Dict[str, Deque[Dict[str, float]]] = defaultdict(lambda: deque(maxlen=1200))

    def compute(self, snapshot: MarketSnapshot, timeframe: str = "1min") -> FeaturePacket:
        df = snapshot.frames[timeframe].copy()
        close = pd.to_numeric(df["close"], errors="coerce").ffill()
        high = pd.to_numeric(df["high"], errors="coerce").ffill()
        low = pd.to_numeric(df["low"], errors="coerce").ffill()
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

        returns = close.pct_change().fillna(0.0)
        atr = (high - low).rolling(14).mean().bfill().fillna(0.0)
        ema_fast = _ema(close, 12)
        ema_slow = _ema(close, 26)
        trend_strength = _safe_float((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / max(close.iloc[-1], 1e-8))
        momentum_1 = _safe_float(returns.iloc[-1])
        momentum_5 = _safe_float(close.pct_change(5).iloc[-1])
        momentum_15 = _safe_float(close.pct_change(15).iloc[-1])
        volatility_10 = _safe_float(returns.tail(10).std())
        volatility_30 = _safe_float(returns.tail(30).std())
        rsi_14 = _safe_float(_rsi(close, 14).iloc[-1]) / 100.0
        vwap = _safe_float((close * volume).sum() / max(volume.sum(), 1e-9), _safe_float(close.iloc[-1]))
        vwap_deviation = _safe_float((close.iloc[-1] - vwap) / max(vwap, 1e-9))
        volume_spike = _safe_float(
            volume.tail(3).mean() / max(_safe_float(volume.tail(30).mean()), 1e-9)
        )

        raw = {
            "momentum_1": momentum_1,
            "momentum_5": momentum_5,
            "momentum_15": momentum_15,
            "volatility_10": volatility_10,
            "volatility_30": volatility_30,
            "trend_strength": trend_strength,
            "rsi_14": rsi_14,
            "atr_pct": _safe_float(atr.iloc[-1] / max(close.iloc[-1], 1e-9)),
            "orderbook_imbalance": _safe_float(snapshot.orderbook_imbalance),
            "volume_imbalance": _safe_float(snapshot.volume_imbalance),
            "volume_spike": volume_spike,
            "vwap_deviation": vwap_deviation,
            "funding_rate": _safe_float(snapshot.funding_rate),
            "spread_pct": _safe_float(snapshot.spread_pct),
            "realized_volatility": _safe_float(snapshot.realized_volatility),
        }
        key = f"{snapshot.symbol}:{timeframe}"
        self._history[key].append(raw)
        normalized = self._normalize(key, raw)
        return FeaturePacket(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            timeframe=timeframe,
            raw_features=raw,
            normalized_features=normalized,
        )

    def _normalize(self, key: str, raw: Dict[str, float]) -> Dict[str, float]:
        hist = list(self._history[key])
        frame = pd.DataFrame(hist).fillna(0.0)
        if len(frame) < 20:
            return dict(raw)
        scaler = self._scalers.get(key)
        if scaler is None:
            scaler = StandardScaler()
            self._scalers[key] = scaler
            scaler.fit(frame.iloc[:-1].values)
        else:
            scaler.partial_fit(frame.iloc[-20:].values)
        values = scaler.transform(frame.tail(1).values)[0]
        columns = list(frame.columns)
        return {columns[idx]: _safe_float(values[idx]) for idx in range(len(columns))}

    def feature_history(self, symbol: str, timeframe: str = "1min") -> pd.DataFrame:
        key = f"{symbol}:{timeframe}"
        return pd.DataFrame(list(self._history[key]))
