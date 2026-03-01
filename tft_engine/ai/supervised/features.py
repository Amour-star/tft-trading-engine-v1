"""
Feature engineering for supervised trade prediction.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "rsi",
    "ema_20",
    "ema_50",
    "atr",
    "volume_change",
    "volatility",
    "macd",
    "market_regime_encoded",
    "hour",
    "day_of_week",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0.0).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def _macd(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    return ema_fast - ema_slow


def _encode_market_regime(regime: str) -> int:
    mapping = {"bull": 2, "sideways": 1, "bear": 0}
    return mapping.get((regime or "sideways").lower(), 1)


def detect_market_regime(ema_20: float, ema_50: float, macd: float) -> str:
    if ema_20 > ema_50 and macd > 0:
        return "bull"
    if ema_20 < ema_50 and macd < 0:
        return "bear"
    return "sideways"


def build_feature_matrix(ohlcv: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(set(ohlcv.columns))
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    df = ohlcv.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = _rsi(df["close"])
    df["atr"] = _atr(df)
    df["volatility"] = df["close"].pct_change().rolling(20, min_periods=20).std()
    df["volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["macd"] = _macd(df["close"])
    df["market_regime"] = df.apply(
        lambda row: detect_market_regime(
            float(row["ema_20"]),
            float(row["ema_50"]),
            float(row["macd"]),
        ),
        axis=1,
    )
    df["market_regime_encoded"] = df["market_regime"].map(_encode_market_regime)
    df["hour"] = df["timestamp"].dt.hour.astype(float)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(float)
    return df


def features_from_snapshot(snapshot: dict) -> dict:
    regime = snapshot.get("market_regime")
    if regime is None:
        regime = detect_market_regime(
            float(snapshot.get("ema_20", 0.0)),
            float(snapshot.get("ema_50", 0.0)),
            float(snapshot.get("macd", 0.0)),
        )
    return {
        "rsi": float(snapshot.get("rsi", 50.0)),
        "ema_20": float(snapshot.get("ema_20", snapshot.get("current_price", 0.0))),
        "ema_50": float(snapshot.get("ema_50", snapshot.get("current_price", 0.0))),
        "atr": float(snapshot.get("atr", 0.0)),
        "volume_change": float(snapshot.get("volume_change", 0.0)),
        "volatility": float(snapshot.get("volatility", 0.0)),
        "macd": float(snapshot.get("macd", 0.0)),
        "market_regime_encoded": float(_encode_market_regime(regime)),
        "hour": float(snapshot.get("hour", 0.0)),
        "day_of_week": float(snapshot.get("day_of_week", 0.0)),
    }


def feature_vector(features: dict, columns: Iterable[str] = FEATURE_COLUMNS) -> np.ndarray:
    return np.asarray([float(features.get(c, 0.0)) for c in columns], dtype=float)

