"""
Feature engineering for TFT model.
Computes technical indicators, market regime classification, and temporal features.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


def compute_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute all features for TFT input.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
    btc_df : pd.DataFrame, optional
        Reference-asset OHLCV for correlation features

    Returns
    -------
    pd.DataFrame with original + feature columns
    """
    df = df.copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---- Price-based ----
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # ---- ATR (Average True Range) ----
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_7"] = true_range.rolling(7).mean()

    # ---- RSI ----
    df["rsi_14"] = _compute_rsi(df["close"], 14)
    df["rsi_7"] = _compute_rsi(df["close"], 7)

    # ---- EMA ----
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # ---- EMA crossover signals ----
    df["ema_9_21_diff"] = (df["ema_9"] - df["ema_21"]) / df["close"]
    df["ema_21_50_diff"] = (df["ema_21"] - df["ema_50"]) / df["close"]

    # ---- Volume features ----
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    df["volume_delta"] = df["volume"].diff()

    # Buy/sell volume approximation
    df["buy_volume"] = df["volume"] * (
        (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, 1)
    )
    df["sell_volume"] = df["volume"] - df["buy_volume"]
    df["volume_imbalance"] = (df["buy_volume"] - df["sell_volume"]) / df["volume"].replace(0, 1)

    # ---- Volatility features ----
    df["volatility_20"] = df["returns"].rolling(20).std()
    df["volatility_60"] = df["returns"].rolling(60).std()
    df["volatility_ratio"] = df["volatility_20"] / df["volatility_60"].replace(0, np.nan)

    # ---- Volatility regime ----
    vol_median = df["volatility_20"].rolling(100).median()
    df["volatility_regime"] = pd.cut(
        df["volatility_20"] / vol_median.replace(0, np.nan),
        bins=[-np.inf, 0.5, 1.0, 1.5, np.inf],
        labels=["low", "normal", "high", "extreme"],
    ).astype(str)

    # ---- Market regime (trend classification) ----
    df["market_regime"] = _classify_market_regime(df)

    # ---- Temporal features ----
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Session encoding (UTC-based)
        df["session"] = df["hour"].apply(_get_session)
    else:
        for col in ["hour", "day_of_week", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            df[col] = 0
        df["session"] = "unknown"

    # ---- BTC correlation / dominance ----
    if btc_df is not None and not btc_df.empty:
        btc_df = btc_df.copy()
        btc_df["btc_returns"] = btc_df["close"].pct_change()
        merged = df.merge(
            btc_df[["timestamp", "btc_returns", "close"]].rename(columns={"close": "btc_close"}),
            on="timestamp",
            how="left",
        )
        df["btc_returns"] = merged["btc_returns"].values
        df["btc_correlation"] = (
            df["returns"].rolling(20).corr(df["btc_returns"])
        )
        df["btc_close"] = merged["btc_close"].values
    else:
        df["btc_returns"] = 0.0
        df["btc_correlation"] = 0.0
        df["btc_close"] = 0.0

    # ---- Momentum ----
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_20"] = df["close"].pct_change(20)

    # ---- Bollinger Bands ----
    bb_sma = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = (bb_sma + 2 * bb_std)
    df["bb_lower"] = (bb_sma - 2 * bb_std)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_sma
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, 1)

    # ---- MACD ----
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ---- Final NaN / inf safety for model compatibility ----
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
    categorical_cols = [
        c for c in ["volatility_regime", "market_regime", "session"] if c in df.columns
    ]
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown").astype(str)

    logger.debug(f"Computed {len(df.columns)} features for {len(df)} rows")
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _classify_market_regime(df: pd.DataFrame) -> pd.Series:
    """Classify market regime based on EMA alignment and momentum."""
    regimes = []
    for i in range(len(df)):
        if i < 200:
            regimes.append("unknown")
            continue
        ema_9 = df["ema_9"].iloc[i]
        ema_21 = df["ema_21"].iloc[i]
        ema_50 = df["ema_50"].iloc[i]
        ema_200 = df["ema_200"].iloc[i]

        if ema_9 > ema_21 > ema_50 > ema_200:
            regimes.append("strong_uptrend")
        elif ema_9 > ema_21 > ema_50:
            regimes.append("uptrend")
        elif ema_9 < ema_21 < ema_50 < ema_200:
            regimes.append("strong_downtrend")
        elif ema_9 < ema_21 < ema_50:
            regimes.append("downtrend")
        else:
            regimes.append("ranging")
    return pd.Series(regimes, index=df.index)


def _get_session(hour: int) -> str:
    """Map UTC hour to trading session."""
    if 0 <= hour < 8:
        return "asia"
    elif 8 <= hour < 14:
        return "europe"
    elif 14 <= hour < 21:
        return "us"
    else:
        return "late_us"


# ---------------------------------------------------------------------------
# Multi-timeframe feature alignment
# ---------------------------------------------------------------------------

def align_multi_timeframe(
    dfs: dict[str, pd.DataFrame],
    base_timeframe: str = "15min",
) -> pd.DataFrame:
    """
    Merge features from multiple timeframes onto the base timeframe.
    Higher timeframes are forward-filled to align with base candles.
    """
    base = dfs[base_timeframe].copy()
    base.set_index("timestamp", inplace=True)

    for tf, df in dfs.items():
        if tf == base_timeframe:
            continue
        suffix = f"_{tf}"
        higher = df.copy()
        higher.set_index("timestamp", inplace=True)
        # Select key features from higher TF
        cols_to_merge = [
            "atr_14", "rsi_14", "ema_9_21_diff", "volatility_20",
            "volume_ratio", "momentum_10", "market_regime", "volatility_regime",
        ]
        available = [c for c in cols_to_merge if c in higher.columns]
        higher = higher[available].add_suffix(suffix)
        base = base.join(higher, how="left")
        # Forward fill higher TF data
        for col in higher.columns:
            base[col] = base[col].ffill()

    base.reset_index(inplace=True)
    base.ffill(inplace=True)
    base.bfill(inplace=True)
    return base


def get_feature_columns() -> list[str]:
    """Return the list of feature columns used by the TFT model."""
    return [
        "returns", "log_returns",
        "atr_14", "atr_7",
        "rsi_14", "rsi_7",
        "ema_9_21_diff", "ema_21_50_diff",
        "volume_ratio", "volume_delta", "volume_imbalance",
        "volatility_20", "volatility_60", "volatility_ratio",
        "momentum_5", "momentum_10", "momentum_20",
        "bb_width", "bb_position",
        "macd", "macd_signal", "macd_hist",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "btc_returns", "btc_correlation",
    ]


def get_categorical_columns() -> list[str]:
    """Return categorical feature columns."""
    return ["volatility_regime", "market_regime", "session"]
