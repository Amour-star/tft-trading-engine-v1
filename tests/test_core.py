"""
Unit tests for core components.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestFeatureEngineering:
    """Test feature computation."""

    def _make_ohlcv(self, n: int = 300) -> pd.DataFrame:
        """Create synthetic OHLCV data."""
        np.random.seed(42)
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=15 * i) for i in range(n)]
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.1
        volume = np.random.exponential(1000, n)
        return pd.DataFrame({
            "timestamp": timestamps, "open": open_, "high": high,
            "low": low, "close": close, "volume": volume,
        })

    def test_compute_features_shape(self):
        from data.features import compute_features
        df = self._make_ohlcv()
        result = compute_features(df)
        assert len(result) == len(df)
        assert "atr_14" in result.columns
        assert "rsi_14" in result.columns
        assert "ema_9" in result.columns
        assert "volatility_regime" in result.columns
        assert "market_regime" in result.columns
        assert "hour_sin" in result.columns

    def test_rsi_range(self):
        from data.features import compute_features
        df = self._make_ohlcv()
        result = compute_features(df)
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_feature_columns(self):
        from data.features import get_feature_columns, get_categorical_columns
        features = get_feature_columns()
        assert len(features) > 10
        cats = get_categorical_columns()
        assert "volatility_regime" in cats


class TestBacktester:
    """Test backtesting engine."""

    def test_backtest_basic(self):
        from backtesting.backtester import Backtester
        np.random.seed(42)
        n = 500
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": close + np.random.randn(n) * 0.1,
            "high": close + np.abs(np.random.randn(n) * 0.3),
            "low": close - np.abs(np.random.randn(n) * 0.3),
            "close": close,
            "volume": np.random.exponential(1000, n),
        })

        signals = [{
            "timestamp": timestamps[50],
            "pair": "TEST-USDT",
            "entry_price": float(close[50]),
            "stop_price": float(close[50] * 0.98),
            "target_price": float(close[50] * 1.04),
            "confidence": 0.75,
        }]

        bt = Backtester(initial_balance=10000)
        result = bt.run(df, signals)
        assert result.total_trades >= 1
        assert len(result.equity_curve) > 1

    def test_monte_carlo(self):
        from backtesting.backtester import Backtester, BacktestTrade
        trades = [
            BacktestTrade(pair="T", side="BUY", entry_time=datetime.now(),
                          entry_price=100, stop_price=98, target_price=104,
                          pnl=10 * (1 if i % 3 != 0 else -1), pnl_pct=0.01, r_multiple=1.0)
            for i in range(30)
        ]
        bt = Backtester()
        result = bt.monte_carlo(trades, n_simulations=100)
        assert "median_final_balance" in result
        assert "probability_profit" in result


class TestSafety:
    """Test safety layer logic."""

    def test_safety_defaults(self):
        from engine.safety import SafetyManager
        sm = SafetyManager()
        assert not sm.is_paused
        assert not sm.is_killed

    def test_kill_switch(self):
        from engine.safety import SafetyManager
        sm = SafetyManager()
        sm._killed = True
        can, reason = sm.can_trade()
        assert not can
        assert "kill" in reason.lower()


class TestExecution:
    """Test execution engine utilities."""

    def test_round_price(self):
        from execution.executor import ExecutionEngine
        # Test with mock - just verify the rounding logic
        import math
        increment = 0.01
        price = 123.456789
        decimals = max(0, -int(math.floor(math.log10(increment))))
        rounded = round(math.floor(price / increment) * increment, decimals)
        assert rounded == 123.45

    def test_position_sizing_logic(self):
        balance = 10000
        risk_pct = 0.01
        entry = 100.0
        stop = 98.0
        risk_amount = balance * risk_pct  # 100
        stop_distance = abs(entry - stop)  # 2
        qty = risk_amount / stop_distance  # 50
        assert qty == 50.0
        assert qty * entry == 5000  # 50% of balance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
