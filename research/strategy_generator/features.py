from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def _safe_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period, min_periods=1).mean()


class StrategyFeatureBuilder:
    """Builds shared strategy and ML features from OHLCV or microstructure frames."""

    def __init__(
        self,
        volatility_window: int = 20,
        vwap_window: int = 20,
        momentum_windows: Tuple[int, ...] = (3, 5, 10),
    ) -> None:
        self.volatility_window = max(5, int(volatility_window))
        self.vwap_window = max(5, int(vwap_window))
        self.momentum_windows = tuple(sorted({max(1, int(item)) for item in momentum_windows}))

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame.sort_values("timestamp", inplace=True)
        frame.reset_index(drop=True, inplace=True)

        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = _safe_series(frame[col]).ffill().bfill()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame.dropna(subset=["timestamp", "open", "high", "low", "close"], inplace=True)
        frame.reset_index(drop=True, inplace=True)

        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        volume = _safe_series(frame["volume"]).fillna(0.0)

        typical = (high + low + close) / 3.0
        rolling_turnover = (typical * volume).rolling(self.vwap_window, min_periods=1).sum()
        rolling_volume = volume.rolling(self.vwap_window, min_periods=1).sum().replace(0.0, np.nan)
        vwap = (rolling_turnover / rolling_volume).fillna(close)

        candle_range = (high - low).replace(0.0, np.nan)
        buy_volume = volume * ((close - low) / candle_range).clip(lower=0.0, upper=1.0).fillna(0.5)
        sell_volume = (volume - buy_volume).clip(lower=0.0)
        if {"bid_volume", "ask_volume"}.issubset(frame.columns):
            bid_volume = _safe_series(frame["bid_volume"]).fillna(buy_volume)
            ask_volume = _safe_series(frame["ask_volume"]).fillna(sell_volume)
        else:
            bid_volume = buy_volume
            ask_volume = sell_volume

        liquidity_den = (bid_volume + ask_volume).replace(0.0, np.nan)
        liquidity_imbalance = ((bid_volume - ask_volume) / liquidity_den).fillna(0.0)

        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        volume_mean = volume.rolling(self.volatility_window, min_periods=1).mean().replace(0.0, np.nan)

        frame["returns"] = returns
        frame["rolling_volatility"] = returns.rolling(self.volatility_window, min_periods=2).std().fillna(0.0)
        frame["vwap"] = vwap
        frame["vwap_deviation"] = ((close - vwap) / vwap.replace(0.0, np.nan)).fillna(0.0)
        frame["liquidity_imbalance"] = liquidity_imbalance
        frame["orderbook_imbalance"] = liquidity_imbalance
        frame["volume_delta"] = (buy_volume - sell_volume).fillna(0.0)
        frame["volume_spike"] = (volume / volume_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        frame["atr_14"] = _atr(high, low, close, 14)
        frame["atr_pct"] = (frame["atr_14"] / close.replace(0.0, np.nan)).fillna(0.0)
        frame["rsi_14"] = _rsi(close, 14)
        frame["ema_fast"] = _ema(close, 12)
        frame["ema_slow"] = _ema(close, 26)
        frame["ema_gap"] = ((frame["ema_fast"] - frame["ema_slow"]) / close.replace(0.0, np.nan)).fillna(0.0)

        for window in self.momentum_windows:
            frame[f"momentum_{window}"] = close.pct_change(window).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if "spread_pct" in frame.columns:
            frame["spread_pct"] = _safe_series(frame["spread_pct"]).fillna(0.0)
        else:
            synthetic_spread = ((high - low) / close.replace(0.0, np.nan)).fillna(0.0) * 0.15
            frame["spread_pct"] = synthetic_spread.clip(lower=0.0001, upper=0.01)

        frame["future_return_1"] = close.shift(-1) / close - 1.0
        frame["future_return_4"] = close.shift(-4) / close - 1.0
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        frame.ffill(inplace=True)
        frame.bfill(inplace=True)
        return frame

    def feature_columns(self) -> Iterable[str]:
        cols = [
            "rolling_volatility",
            "vwap_deviation",
            "liquidity_imbalance",
            "volume_delta",
            "volume_spike",
            "atr_pct",
            "rsi_14",
            "ema_gap",
            "orderbook_imbalance",
        ]
        cols.extend([f"momentum_{window}" for window in self.momentum_windows])
        return cols

    def build_model_matrix(
        self,
        df: pd.DataFrame,
        target_horizon: int = 4,
        target_threshold: float = 0.0,
    ) -> tuple[pd.DataFrame, pd.Series]:
        frame = self.build(df)
        X = frame[list(self.feature_columns())].copy()
        y = (frame["close"].shift(-target_horizon) / frame["close"] - 1.0 > target_threshold).astype(int)
        valid = y.notna()
        return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
