"""
Comprehensive test suite for TFT Trading Engine.
Tests the complete lifecycle: signal generation → position opening →
monitoring → position closing → PnL calculation → trade history.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from data.database import Base, Trade, EngineState, DailyStats
from tests.conftest import make_ohlcv, make_mock_fetcher, make_mock_predictor


# =========================================================================
# PHASE 1: Feature Engineering Tests
# =========================================================================

class TestFeatureEngineering:
    """Validate feature computation produces correct outputs."""

    def test_compute_features_all_columns(self):
        """Verify all expected feature columns are present."""
        from data.features import compute_features, get_feature_columns
        df = make_ohlcv(300)
        result = compute_features(df)
        for col in get_feature_columns():
            assert col in result.columns, f"Missing feature column: {col}"

    def test_compute_features_no_nan_after_warmup(self):
        """After warm-up period, features should have no NaN values."""
        from data.features import compute_features
        df = make_ohlcv(300)
        result = compute_features(df)
        # After row 200 (warm-up for EMA-200), there should be no NaN
        tail = result.iloc[201:]
        numeric_cols = tail.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            nan_count = tail[col].isna().sum()
            assert nan_count == 0, f"Column {col} has {nan_count} NaN values after warm-up"

    def test_rsi_bounds(self):
        """RSI should always be between 0 and 100."""
        from data.features import compute_features
        df = make_ohlcv(300)
        result = compute_features(df)
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0, f"RSI below 0: {rsi.min()}"
        assert rsi.max() <= 100, f"RSI above 100: {rsi.max()}"

    def test_volatility_regime_values(self):
        """Volatility regime should only contain valid categories."""
        from data.features import compute_features
        df = make_ohlcv(300)
        result = compute_features(df)
        valid_regimes = {"low", "normal", "high", "extreme", "nan"}
        actual = set(result["volatility_regime"].unique())
        assert actual.issubset(valid_regimes), f"Invalid regimes: {actual - valid_regimes}"

    def test_market_regime_values(self):
        """Market regime should contain valid classifications."""
        from data.features import compute_features
        df = make_ohlcv(300)
        result = compute_features(df)
        valid_regimes = {"unknown", "strong_uptrend", "uptrend", "ranging", "downtrend", "strong_downtrend"}
        actual = set(result["market_regime"].unique())
        assert actual.issubset(valid_regimes), f"Invalid regimes: {actual - valid_regimes}"

    def test_btc_correlation_with_btc_data(self):
        """BTC correlation should be computed when btc_df is provided."""
        from data.features import compute_features
        df = make_ohlcv(300, base_price=100.0, seed=42)
        btc_df = make_ohlcv(300, base_price=50000.0, seed=123)
        result = compute_features(df, btc_df)
        assert "btc_correlation" in result.columns
        # Should have non-zero values (at least after warm-up)
        assert result["btc_correlation"].iloc[50:].abs().sum() > 0

    def test_temporal_features(self):
        """Temporal features (hour_sin, hour_cos, etc.) should be present."""
        from data.features import compute_features
        df = make_ohlcv(300)
        result = compute_features(df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "session"]:
            assert col in result.columns, f"Missing temporal feature: {col}"
        # sin/cos should be in [-1, 1]
        assert result["hour_sin"].min() >= -1
        assert result["hour_sin"].max() <= 1


# =========================================================================
# PHASE 2: Paper Executor Tests
# =========================================================================

class TestPaperExecutor:
    """Test paper trading executor with persistent SQLite state."""

    def _create_paper_executor(self, fetcher=None, balance=10000.0):
        """Helper to create a paper executor with temp database."""
        from execution.paper_executor import PaperExecutor
        f = fetcher or make_mock_fetcher(current_price=100.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        executor = PaperExecutor(
            fetcher=f,
            starting_balance=balance,
            fee_rate=0.001,
            slippage_bps=0.0,
            db_path=tmp.name,
        )
        return executor, tmp.name

    def test_initial_balance(self):
        """Executor should start with the specified balance."""
        executor, db_path = self._create_paper_executor(balance=10000.0)
        assert executor.get_balance() == 10000.0
        os.unlink(db_path)

    def test_buy_reduces_balance(self):
        """Buying should reduce balance by cost + fees."""
        executor, db_path = self._create_paper_executor(balance=10000.0)
        result = executor.buy("BTC-USDT", 1.0, price=100.0)
        assert result["status"] == "filled"
        assert result["fill_price"] == 100.0
        # Cost = 100 * 1.0 = 100, Fee = 100 * 1.0 * 0.001 = 0.1
        expected_balance = 10000.0 - 100.0 - 0.1
        assert abs(executor.get_balance() - expected_balance) < 0.01
        os.unlink(db_path)

    def test_sell_increases_balance(self):
        """Selling should increase balance and calculate realized PnL."""
        fetcher = make_mock_fetcher(current_price=110.0)
        executor, db_path = self._create_paper_executor(fetcher=fetcher, balance=10000.0)
        # Buy first
        executor.buy("BTC-USDT", 1.0, price=100.0)
        # Sell at higher price
        result = executor.sell("BTC-USDT", 1.0, price=110.0)
        assert result["status"] == "filled"
        assert result["realized_pnl"] is not None
        # PnL = proceeds - cost = (110*1 - 0.11) - (100*1) = 9.89
        # (proceeds = 110*1 - fee, fee = 110*1*0.001 = 0.11)
        assert result["realized_pnl"] > 0
        os.unlink(db_path)

    def test_sell_without_position_raises(self):
        """Selling without a position should raise ValueError."""
        executor, db_path = self._create_paper_executor()
        with pytest.raises(ValueError, match="No open paper position"):
            executor.sell("BTC-USDT", 1.0, price=100.0)
        os.unlink(db_path)

    def test_position_tracking(self):
        """Positions should be tracked correctly."""
        executor, db_path = self._create_paper_executor()
        executor.buy("BTC-USDT", 1.0, price=100.0)
        positions = executor.get_positions()
        assert "BTC-USDT" in positions
        assert positions["BTC-USDT"]["quantity"] == 1.0
        assert positions["BTC-USDT"]["avg_entry_price"] == 100.0
        os.unlink(db_path)

    def test_close_position(self):
        """Close position should sell entire position."""
        executor, db_path = self._create_paper_executor()
        executor.buy("BTC-USDT", 2.0, price=100.0)
        result = executor.close_position("BTC-USDT")
        assert result["status"] == "filled"
        positions = executor.get_positions()
        assert "BTC-USDT" not in positions
        os.unlink(db_path)

    def test_persistence(self):
        """State should persist across executor instances."""
        from execution.paper_executor import PaperExecutor
        fetcher = make_mock_fetcher(current_price=100.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()

        # Create executor and buy
        exec1 = PaperExecutor(fetcher=fetcher, starting_balance=10000.0, db_path=tmp.name)
        exec1.buy("BTC-USDT", 1.0, price=100.0)
        balance_after_buy = exec1.get_balance()

        # Create new executor from same database
        exec2 = PaperExecutor(fetcher=fetcher, starting_balance=10000.0, db_path=tmp.name)
        assert abs(exec2.get_balance() - balance_after_buy) < 0.01
        positions = exec2.get_positions()
        assert "BTC-USDT" in positions
        os.unlink(tmp.name)

    def test_trade_history(self):
        """Trade history should record all buy/sell events."""
        executor, db_path = self._create_paper_executor()
        executor.buy("BTC-USDT", 1.0, price=100.0)
        executor.sell("BTC-USDT", 1.0, price=110.0)
        assert len(executor.trade_history) == 2
        assert executor.trade_history[0]["side"] == "BUY"
        assert executor.trade_history[1]["side"] == "SELL"
        os.unlink(db_path)

    def test_insufficient_balance(self):
        """Buying beyond balance should raise ValueError."""
        executor, db_path = self._create_paper_executor(balance=100.0)
        with pytest.raises(ValueError, match="Insufficient paper balance"):
            executor.buy("BTC-USDT", 10.0, price=100.0)  # Cost = 1000 > 100
        os.unlink(db_path)


# =========================================================================
# PHASE 3: Decision Engine Tests
# =========================================================================

class TestDecisionEngine:
    """Test signal generation and pair evaluation."""

    def test_generate_signal_bullish(self):
        """Should generate a BUY signal when prediction is bullish."""
        from engine.decision import DecisionEngine
        fetcher = make_mock_fetcher(current_price=100.0)
        predictor = make_mock_predictor(prob_up=0.7, prob_down=0.3, confidence=0.75)
        engine = DecisionEngine(fetcher, predictor)
        engine.confidence_threshold = 0.5
        signal = engine.generate_signal()
        assert signal is not None
        assert signal.side == "BUY"
        assert signal.entry_price > 0
        assert signal.stop_price < signal.entry_price
        assert signal.target_price > signal.entry_price
        assert signal.risk_reward >= 2.0

    def test_generate_signal_bearish_returns_none(self):
        """Should return None when prediction is bearish (spot-only)."""
        from engine.decision import DecisionEngine
        fetcher = make_mock_fetcher(current_price=100.0)
        predictor = make_mock_predictor(prob_up=0.3, prob_down=0.7, confidence=0.75)
        engine = DecisionEngine(fetcher, predictor)
        signal = engine.generate_signal()
        assert signal is None

    def test_low_confidence_filtered(self):
        """Low-confidence predictions should be filtered out."""
        from engine.decision import DecisionEngine
        fetcher = make_mock_fetcher(current_price=100.0)
        predictor = make_mock_predictor(prob_up=0.7, prob_down=0.3, confidence=0.3)
        engine = DecisionEngine(fetcher, predictor)
        engine.confidence_threshold = 0.55
        signal = engine.generate_signal()
        assert signal is None

    def test_signal_has_correct_structure(self):
        """Signal should have all required fields."""
        from engine.decision import DecisionEngine, TradeSignal
        fetcher = make_mock_fetcher(current_price=100.0)
        predictor = make_mock_predictor(prob_up=0.7, prob_down=0.3, confidence=0.75)
        engine = DecisionEngine(fetcher, predictor)
        engine.confidence_threshold = 0.5
        signal = engine.generate_signal()
        assert signal is not None
        assert isinstance(signal, TradeSignal)
        assert signal.pair != ""
        assert signal.atr > 0
        assert 0 <= signal.confidence <= 1
        assert signal.reasoning != ""

    def test_atr_based_stop_loss(self):
        """Stop loss should be 2x ATR below entry."""
        from engine.decision import DecisionEngine
        fetcher = make_mock_fetcher(current_price=100.0)
        predictor = make_mock_predictor(prob_up=0.7, prob_down=0.3, confidence=0.75)
        engine = DecisionEngine(fetcher, predictor)
        engine.confidence_threshold = 0.5
        signal = engine.generate_signal()
        assert signal is not None
        # stop_price = entry_price - 2 * ATR
        stop_distance = signal.entry_price - signal.stop_price
        assert stop_distance > 0
        expected_stop_distance = signal.atr * 2.0
        assert abs(stop_distance - expected_stop_distance) < 0.01

    def test_update_thresholds(self):
        """Thresholds should be updatable."""
        from engine.decision import DecisionEngine
        fetcher = make_mock_fetcher()
        predictor = make_mock_predictor()
        engine = DecisionEngine(fetcher, predictor)
        old_threshold = engine.confidence_threshold
        engine.update_thresholds({"confidence_threshold": 0.8})
        assert engine.confidence_threshold == 0.8

    def test_threshold_bounds(self):
        """Thresholds should be clamped to valid ranges."""
        from engine.decision import DecisionEngine
        fetcher = make_mock_fetcher()
        predictor = make_mock_predictor()
        engine = DecisionEngine(fetcher, predictor)
        engine.update_thresholds({"confidence_threshold": 0.1})
        assert engine.confidence_threshold >= 0.5  # Clamped to min
        engine.update_thresholds({"confidence_threshold": 0.99})
        assert engine.confidence_threshold <= 0.95  # Clamped to max


# =========================================================================
# PHASE 4: Position Opening via execute_signal
# =========================================================================

class TestPositionOpening:
    """Test the full signal → execute → record flow."""

    def test_execute_signal_opens_position(self, patch_db):
        """Executing a signal should open a position in the database."""
        from engine.decision import TradeSignal
        from execution.paper_executor import PaperExecutor

        fetcher = make_mock_fetcher(current_price=100.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        executor = PaperExecutor(fetcher=fetcher, starting_balance=10000.0, db_path=tmp.name)

        signal = TradeSignal(
            pair="BTC-USDT", side="BUY", entry_price=100.01,
            stop_price=98.0, target_price=104.0, confidence=0.75,
            expected_move=0.02, prob_up=0.7, prob_down=0.3,
            risk_reward=2.0, atr=1.0, volatility_regime="normal",
            market_regime="uptrend", btc_correlation=0.5,
            spread_pct=0.001, volume_24h=1000000.0,
            reasoning="Test signal",
        )

        trade_id = executor.execute_signal(signal, balance=10000.0)
        assert trade_id is not None

        # Verify trade recorded in database
        session = patch_db()
        trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
        assert trade is not None
        assert trade.status == "open"
        assert trade.pair == "BTC-USDT"
        assert trade.entry_price > 0
        assert trade.stop_price == 98.0
        assert trade.target_price == 104.0
        session.close()
        os.unlink(tmp.name)

    def test_position_size_calculation(self):
        """Position size should respect risk parameters."""
        from engine.decision import TradeSignal
        from execution.paper_executor import PaperExecutor

        fetcher = make_mock_fetcher(current_price=100.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        executor = PaperExecutor(fetcher=fetcher, starting_balance=10000.0, db_path=tmp.name)

        signal = TradeSignal(
            pair="BTC-USDT", side="BUY", entry_price=100.0,
            stop_price=98.0, target_price=104.0, confidence=0.75,
            expected_move=0.02, prob_up=0.7, prob_down=0.3,
            risk_reward=2.0, atr=1.0, volatility_regime="normal",
            market_regime="uptrend", btc_correlation=0.5,
            spread_pct=0.001, volume_24h=1000000.0,
            reasoning="Test signal",
        )

        # risk_pct = confidence * risk_per_trade = 0.75 * 0.01 = 0.0075
        # risk_amount = 10000 * 0.0075 = 75
        # stop_distance = |100 - 98| = 2
        # qty = 75 / 2 = 37.5
        qty = executor.calculate_position_size(signal, 10000.0, 0.0075)
        assert qty > 0
        assert qty <= 10000.0 / 100.0  # Can't exceed balance
        os.unlink(tmp.name)


# =========================================================================
# PHASE 5: Position Closing Tests (CRITICAL)
# =========================================================================

class TestPositionClosing:
    """Test that positions close correctly at TP, SL, and other conditions."""

    def _setup_open_trade(self, session, pair="BTC-USDT", entry_price=100.0,
                          stop_price=98.0, target_price=104.0, qty=1.0):
        """Insert an open trade into the database."""
        trade = Trade(
            trade_id="test_trade_001",
            pair=pair,
            side="BUY",
            entry_time=datetime.utcnow(),
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=qty,
            status="open",
            confidence=0.75,
        )
        session.add(trade)
        session.commit()
        return trade

    def test_close_at_target(self, patch_db):
        """Position should close when price hits target."""
        from execution.monitor import PositionMonitor

        session = patch_db()
        self._setup_open_trade(session, entry_price=100.0, stop_price=98.0,
                               target_price=104.0, qty=1.0)
        session.close()

        # Price hits target
        fetcher = make_mock_fetcher(current_price=104.5)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        from execution.paper_executor import PaperExecutor
        executor = PaperExecutor(fetcher=fetcher, starting_balance=10000.0, db_path=tmp.name)
        # Pre-populate paper position so sell doesn't fail
        executor._positions["BTC-USDT"] = {"quantity": 1.0, "avg_entry_price": 100.0}
        executor._persist_state()

        predictor = make_mock_predictor()
        monitor = PositionMonitor(fetcher, executor, predictor)

        trade_id = monitor.monitor_cycle()
        assert trade_id == "test_trade_001"

        # Verify trade is closed in database
        session = patch_db()
        trade = session.query(Trade).filter(Trade.trade_id == "test_trade_001").first()
        assert trade.status == "closed"
        assert trade.exit_reason == "target"
        assert trade.pnl is not None
        assert trade.pnl > 0  # Profitable
        assert trade.exit_price > 0
        session.close()
        os.unlink(tmp.name)

    def test_close_at_stop_loss(self, patch_db):
        """Position should close when price hits stop loss."""
        from execution.monitor import PositionMonitor

        session = patch_db()
        self._setup_open_trade(session, entry_price=100.0, stop_price=98.0,
                               target_price=104.0, qty=1.0)
        session.close()

        # Price drops below stop
        fetcher = make_mock_fetcher(current_price=97.5)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        from execution.paper_executor import PaperExecutor
        executor = PaperExecutor(fetcher=fetcher, starting_balance=10000.0, db_path=tmp.name)
        executor._positions["BTC-USDT"] = {"quantity": 1.0, "avg_entry_price": 100.0}
        executor._persist_state()

        predictor = make_mock_predictor()
        monitor = PositionMonitor(fetcher, executor, predictor)

        trade_id = monitor.monitor_cycle()
        assert trade_id == "test_trade_001"

        session = patch_db()
        trade = session.query(Trade).filter(Trade.trade_id == "test_trade_001").first()
        assert trade.status == "closed"
        assert trade.exit_reason == "stop"
        assert trade.pnl is not None
        assert trade.pnl < 0  # Loss
        session.close()
        os.unlink(tmp.name)

    def test_no_close_when_price_between_stops(self, patch_db):
        """No close when price is between stop and target."""
        from execution.monitor import PositionMonitor

        session = patch_db()
        self._setup_open_trade(session, entry_price=100.0, stop_price=98.0,
                               target_price=104.0, qty=1.0)
        session.close()

        # Price between stop and target
        fetcher = make_mock_fetcher(current_price=101.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        from execution.paper_executor import PaperExecutor
        executor = PaperExecutor(fetcher=fetcher, starting_balance=10000.0, db_path=tmp.name)
        executor._positions["BTC-USDT"] = {"quantity": 1.0, "avg_entry_price": 100.0}
        executor._persist_state()

        # Mock predictor to avoid model re-evaluation triggering close
        predictor = make_mock_predictor(prob_up=0.7, prob_down=0.3, confidence=0.75)
        monitor = PositionMonitor(fetcher, executor, predictor)

        trade_id = monitor.monitor_cycle()
        assert trade_id is None

        session = patch_db()
        trade = session.query(Trade).filter(Trade.trade_id == "test_trade_001").first()
        assert trade.status == "open"
        session.close()
        os.unlink(tmp.name)

    def test_pnl_calculation_accuracy(self, patch_db):
        """PnL should be correctly calculated on close."""
        from execution.monitor import PositionMonitor

        session = patch_db()
        self._setup_open_trade(session, entry_price=100.0, stop_price=98.0,
                               target_price=104.0, qty=2.0)
        session.close()

        # Price hits target
        fetcher = make_mock_fetcher(current_price=104.5)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        from execution.paper_executor import PaperExecutor
        executor = PaperExecutor(
            fetcher=fetcher, starting_balance=10000.0, fee_rate=0.001, db_path=tmp.name
        )
        executor._positions["BTC-USDT"] = {"quantity": 2.0, "avg_entry_price": 100.0}
        executor._persist_state()

        predictor = make_mock_predictor()
        monitor = PositionMonitor(fetcher, executor, predictor)
        monitor.monitor_cycle()

        session = patch_db()
        trade = session.query(Trade).filter(Trade.trade_id == "test_trade_001").first()
        assert trade.status == "closed"
        # PnL percent should be positive (exit > entry)
        assert trade.pnl_pct is not None
        assert trade.pnl_pct > 0
        # R multiple should be positive
        assert trade.r_multiple is not None
        assert trade.r_multiple > 0
        session.close()
        os.unlink(tmp.name)

    def test_signal_reversal_close(self, patch_db):
        """Strong bearish signal should force close."""
        from execution.monitor import PositionMonitor

        session = patch_db()
        self._setup_open_trade(session, entry_price=100.0, stop_price=98.0,
                               target_price=104.0, qty=1.0)
        session.close()

        # Price between stops but strong bearish signal
        fetcher = make_mock_fetcher(current_price=101.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        from execution.paper_executor import PaperExecutor
        executor = PaperExecutor(fetcher=fetcher, starting_balance=10000.0, db_path=tmp.name)
        executor._positions["BTC-USDT"] = {"quantity": 1.0, "avg_entry_price": 100.0}
        executor._persist_state()

        # Strong bearish signal (prob_down > 0.75, confidence > 0.6)
        predictor = make_mock_predictor(prob_up=0.2, prob_down=0.8, confidence=0.7)
        monitor = PositionMonitor(fetcher, executor, predictor)

        trade_id = monitor.monitor_cycle()
        assert trade_id == "test_trade_001"

        session = patch_db()
        trade = session.query(Trade).filter(Trade.trade_id == "test_trade_001").first()
        assert trade.status == "closed"
        assert trade.exit_reason == "signal_reversal"
        session.close()
        os.unlink(tmp.name)


# =========================================================================
# PHASE 6: Manual Close All
# =========================================================================

class TestManualCloseAll:
    """Test force close all positions."""

    def test_force_close_all(self, patch_db):
        """Force close should close all open positions."""
        from execution.monitor import PositionMonitor

        session = patch_db()
        # Insert multiple open trades
        for i in range(3):
            trade = Trade(
                trade_id=f"test_trade_{i:03d}",
                pair=f"PAIR{i}-USDT",
                side="BUY",
                entry_time=datetime.utcnow(),
                entry_price=100.0,
                stop_price=98.0,
                target_price=104.0,
                quantity=1.0,
                status="open",
                confidence=0.75,
            )
            session.add(trade)
        session.commit()
        session.close()

        fetcher = make_mock_fetcher(current_price=101.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        from execution.paper_executor import PaperExecutor
        executor = PaperExecutor(fetcher=fetcher, starting_balance=10000.0, db_path=tmp.name)
        for i in range(3):
            executor._positions[f"PAIR{i}-USDT"] = {"quantity": 1.0, "avg_entry_price": 100.0}
        executor._persist_state()

        predictor = make_mock_predictor()
        monitor = PositionMonitor(fetcher, executor, predictor)
        monitor.force_close_all()

        session = patch_db()
        open_trades = session.query(Trade).filter(Trade.status == "open").count()
        closed_trades = session.query(Trade).filter(Trade.status == "closed").count()
        assert open_trades == 0
        assert closed_trades == 3
        for trade in session.query(Trade).filter(Trade.status == "closed").all():
            assert trade.exit_reason == "manual_force_close"
        session.close()
        os.unlink(tmp.name)


# =========================================================================
# PHASE 7: Safety Manager Tests
# =========================================================================

class TestSafetyManager:
    """Test safety controls and circuit breakers."""

    def test_default_state(self):
        """Safety manager should start in a permissive state."""
        from engine.safety import SafetyManager
        sm = SafetyManager()
        assert not sm.is_paused
        assert not sm.is_killed

    def test_kill_switch_blocks_trading(self):
        """Kill switch should block all trading."""
        from engine.safety import SafetyManager
        sm = SafetyManager()
        sm._killed = True
        can, reason = sm.can_trade()
        assert not can
        assert "kill" in reason.lower()

    def test_pause_blocks_trading(self):
        """Paused state should block trading."""
        from engine.safety import SafetyManager
        sm = SafetyManager()
        sm._paused = True
        can, reason = sm.can_trade()
        assert not can
        assert "paused" in reason.lower()

    def test_consecutive_losses_blocks_trading(self, patch_db):
        """Too many consecutive losses should block trading."""
        from engine.safety import SafetyManager
        sm = SafetyManager()
        sm._consecutive_losses = 10  # Above default max of 5
        sm._last_check_date = datetime.now().date()
        can, reason = sm.can_trade()
        assert not can
        assert "consecutive" in reason.lower()

    def test_record_trade_result_updates_stats(self):
        """Recording trade results should update internal state."""
        from engine.safety import SafetyManager
        sm = SafetyManager()
        sm._last_check_date = datetime.now().date()  # Skip DB refresh

        # Mock the _update_daily_stats to avoid DB access
        sm._update_daily_stats = MagicMock()

        sm.record_trade_result(-10.0, "trade_1")
        assert sm._consecutive_losses == 1
        assert sm._daily_pnl == -10.0

        sm.record_trade_result(-5.0, "trade_2")
        assert sm._consecutive_losses == 2
        assert sm._daily_pnl == -15.0

        sm.record_trade_result(20.0, "trade_3")
        assert sm._consecutive_losses == 0  # Reset on win
        assert sm._daily_pnl == 5.0

    def test_volatility_circuit_breaker(self):
        """High volatility should trigger circuit breaker."""
        from engine.safety import SafetyManager
        sm = SafetyManager()
        # 4x baseline (above 3x threshold)
        assert sm.check_volatility_circuit_breaker(0.08, 0.02) is True
        # 2x baseline (below 3x threshold)
        assert sm.check_volatility_circuit_breaker(0.04, 0.02) is False

    def test_api_stability(self):
        """Too many API errors should flag instability."""
        from engine.safety import SafetyManager
        sm = SafetyManager()
        assert sm.check_api_stability(0) is True
        assert sm.check_api_stability(4) is True
        assert sm.check_api_stability(5) is False
        assert sm.check_api_stability(10) is False

    def test_pause_resume(self):
        """Pause and resume should toggle state."""
        from engine.safety import SafetyManager
        sm = SafetyManager()
        sm._save_state = MagicMock()
        sm.pause_trading()
        assert sm.is_paused
        sm.resume_trading()
        assert not sm.is_paused


# =========================================================================
# PHASE 8: Feedback Loop Tests
# =========================================================================

class TestFeedbackLoop:
    """Test the self-improving feedback system."""

    def test_analyze_winning_trade(self, patch_db):
        """Winning trade analysis should identify correct direction."""
        from engine.feedback import FeedbackLoop

        session = patch_db()
        trade = Trade(
            trade_id="win_001",
            pair="BTC-USDT",
            side="BUY",
            entry_time=datetime.utcnow() - timedelta(hours=1),
            exit_time=datetime.utcnow(),
            entry_price=100.0,
            exit_price=104.0,
            stop_price=98.0,
            target_price=104.0,
            quantity=1.0,
            pnl=4.0,
            pnl_pct=0.04,
            r_multiple=2.0,
            exit_reason="target",
            status="closed",
            confidence=0.75,
            prediction={"prob_up": 0.7, "prob_down": 0.3, "expected_move": 0.02},
            features_at_entry={"volatility_regime": "normal"},
        )
        session.add(trade)
        session.commit()
        session.close()

        fl = FeedbackLoop()
        analysis = fl.analyze_trade("win_001")
        assert analysis.get("pnl") == 4.0
        assert analysis.get("direction_correct") is True
        assert analysis.get("volatility_misread") is False

    def test_analyze_losing_trade(self, patch_db):
        """Losing trade should be analyzed for issues."""
        from engine.feedback import FeedbackLoop

        session = patch_db()
        trade = Trade(
            trade_id="loss_001",
            pair="BTC-USDT",
            side="BUY",
            entry_time=datetime.utcnow() - timedelta(hours=1),
            exit_time=datetime.utcnow(),
            entry_price=100.0,
            exit_price=97.5,
            stop_price=98.0,
            target_price=104.0,
            quantity=1.0,
            pnl=-2.5,
            pnl_pct=-0.025,
            r_multiple=-1.25,
            exit_reason="stop",
            status="closed",
            confidence=0.85,
            prediction={"prob_up": 0.8, "prob_down": 0.2, "expected_move": 0.03},
            features_at_entry={"volatility_regime": "normal"},
        )
        session.add(trade)
        session.commit()
        session.close()

        fl = FeedbackLoop()
        analysis = fl.analyze_trade("loss_001")
        assert analysis.get("pnl") == -2.5
        assert analysis.get("confidence_overestimated") is True  # High confidence + loss
        assert analysis.get("volatility_misread") is True  # Normal vol + stop hit + loss

    def test_batch_adjustments_not_enough_trades(self, patch_db):
        """Should return empty if not enough trades."""
        from engine.feedback import FeedbackLoop
        fl = FeedbackLoop()
        result = fl.compute_batch_adjustments()
        assert result == {}


# =========================================================================
# PHASE 9: API Server Tests
# =========================================================================

class TestAPIServer:
    """Test FastAPI dashboard endpoints."""

    @pytest.fixture()
    def api_client_and_db(self, patch_db):
        """Create a test client for the FastAPI app with patched DB."""
        from fastapi.testclient import TestClient
        from dashboard.api import app
        return TestClient(app), patch_db

    def test_status_endpoint(self, api_client_and_db):
        """GET /api/status should return valid status."""
        api_client, _ = api_client_and_db
        response = api_client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "mode" in data
        assert data["mode"] == "PAPER"
        assert "engine_running" in data
        assert "paused" in data
        assert "killed" in data

    def test_trades_endpoint_empty(self, api_client_and_db):
        """GET /api/trades should return empty list initially."""
        api_client, _ = api_client_and_db
        response = api_client.get("/api/trades")
        assert response.status_code == 200
        assert response.json() == []

    def test_trades_endpoint_with_data(self, api_client_and_db):
        """GET /api/trades should return trade records."""
        api_client, get_session = api_client_and_db
        session = get_session()
        trade = Trade(
            trade_id="api_test_001",
            pair="BTC-USDT",
            side="BUY",
            entry_time=datetime.utcnow(),
            entry_price=100.0,
            stop_price=98.0,
            target_price=104.0,
            quantity=1.0,
            status="open",
            confidence=0.75,
        )
        session.add(trade)
        session.commit()
        session.close()

        response = api_client.get("/api/trades")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["trade_id"] == "api_test_001"

    def test_stats_endpoint(self, api_client_and_db):
        """GET /api/stats should return statistics."""
        api_client, _ = api_client_and_db
        response = api_client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_trades" in data
        assert "win_rate" in data
        assert "total_pnl" in data

    def test_control_pause(self, api_client_and_db):
        """POST /api/control with pause should update state."""
        api_client, _ = api_client_and_db
        response = api_client.post("/api/control", json={"action": "pause"})
        assert response.status_code == 200
        assert response.json()["status"] == "paused"

    def test_control_resume(self, api_client_and_db):
        """POST /api/control with resume should update state."""
        api_client, _ = api_client_and_db
        api_client.post("/api/control", json={"action": "pause"})
        response = api_client.post("/api/control", json={"action": "resume"})
        assert response.status_code == 200
        assert response.json()["status"] == "resumed"

    def test_control_kill(self, api_client_and_db):
        """POST /api/control with kill should activate kill switch."""
        api_client, _ = api_client_and_db
        response = api_client.post("/api/control", json={"action": "kill"})
        assert response.status_code == 200
        assert response.json()["status"] == "killed"

    def test_control_invalid_action(self, api_client_and_db):
        """POST /api/control with invalid action should return 400."""
        api_client, _ = api_client_and_db
        response = api_client.post("/api/control", json={"action": "invalid_action"})
        assert response.status_code == 400

    def test_thresholds_endpoint(self, api_client_and_db):
        """GET /api/thresholds should return current thresholds."""
        api_client, _ = api_client_and_db
        response = api_client.get("/api/thresholds")
        assert response.status_code == 200
        data = response.json()
        assert "confidence_threshold" in data

    def test_update_thresholds(self, api_client_and_db):
        """POST /api/thresholds should update thresholds."""
        api_client, _ = api_client_and_db
        response = api_client.post("/api/thresholds", json={
            "confidence_threshold": 0.7,
            "risk_per_trade": 0.015,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["confidence_threshold"] == 0.7
        assert data["risk_per_trade"] == 0.015

    def test_pnl_curve_endpoint(self, api_client_and_db):
        """GET /api/pnl-curve should return PnL curve data."""
        api_client, _ = api_client_and_db
        response = api_client.get("/api/pnl-curve")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_model_info_endpoint(self, api_client_and_db):
        """GET /api/model-info should return model info."""
        api_client, _ = api_client_and_db
        response = api_client.get("/api/model-info")
        assert response.status_code == 200

    def test_learning_metrics_endpoint(self, api_client_and_db):
        """GET /api/learning-metrics should return learning data."""
        api_client, _ = api_client_and_db
        response = api_client.get("/api/learning-metrics")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


# =========================================================================
# PHASE 10: End-to-End Trading Simulation
# =========================================================================

class TestEndToEndSimulation:
    """Full cycle: signal → open → monitor → close → PnL recorded."""

    def test_full_trade_cycle_target_hit(self, patch_db):
        """Complete cycle: open position → price hits target → close → profit recorded."""
        from engine.decision import DecisionEngine, TradeSignal
        from execution.paper_executor import PaperExecutor
        from execution.monitor import PositionMonitor

        # Step 1: Generate signal
        fetcher = make_mock_fetcher(current_price=100.0)
        predictor = make_mock_predictor(prob_up=0.7, prob_down=0.3, confidence=0.75)
        decision = DecisionEngine(fetcher, predictor)
        decision.confidence_threshold = 0.5

        signal = decision.generate_signal()
        assert signal is not None
        assert signal.side == "BUY"

        # Step 2: Execute signal
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        executor = PaperExecutor(fetcher=fetcher, starting_balance=10000.0, db_path=tmp.name)
        trade_id = executor.execute_signal(signal)
        assert trade_id is not None

        # Verify position is open
        session = patch_db()
        trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
        assert trade is not None
        assert trade.status == "open"
        entry_price = trade.entry_price
        target_price = trade.target_price
        session.close()

        # Step 3: Simulate price hitting target
        fetcher_tp = make_mock_fetcher(current_price=target_price + 1.0)
        monitor = PositionMonitor(fetcher_tp, executor, predictor)

        closed_id = monitor.monitor_cycle()
        assert closed_id == trade_id

        # Step 4: Verify trade recorded correctly
        session = patch_db()
        trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
        assert trade.status == "closed"
        assert trade.exit_reason == "target"
        assert trade.pnl is not None
        assert trade.pnl > 0
        assert trade.pnl_pct is not None
        assert trade.pnl_pct > 0
        assert trade.r_multiple is not None
        assert trade.r_multiple > 0
        assert trade.exit_price is not None
        assert trade.exit_time is not None
        session.close()
        os.unlink(tmp.name)

    def test_full_trade_cycle_stop_hit(self, patch_db):
        """Complete cycle: open position → price hits stop → close → loss recorded."""
        from engine.decision import DecisionEngine, TradeSignal
        from execution.paper_executor import PaperExecutor
        from execution.monitor import PositionMonitor

        fetcher = make_mock_fetcher(current_price=100.0)
        predictor = make_mock_predictor(prob_up=0.7, prob_down=0.3, confidence=0.75)
        decision = DecisionEngine(fetcher, predictor)
        decision.confidence_threshold = 0.5

        signal = decision.generate_signal()
        assert signal is not None

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        executor = PaperExecutor(fetcher=fetcher, starting_balance=10000.0, db_path=tmp.name)
        trade_id = executor.execute_signal(signal)
        assert trade_id is not None

        session = patch_db()
        trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
        stop_price = trade.stop_price
        session.close()

        # Simulate price hitting stop
        fetcher_sl = make_mock_fetcher(current_price=stop_price - 1.0)
        monitor = PositionMonitor(fetcher_sl, executor, predictor)
        closed_id = monitor.monitor_cycle()
        assert closed_id == trade_id

        session = patch_db()
        trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
        assert trade.status == "closed"
        assert trade.exit_reason == "stop"
        assert trade.pnl is not None
        assert trade.pnl < 0  # Loss
        session.close()
        os.unlink(tmp.name)

    def test_multiple_trades_history(self, patch_db):
        """Multiple trades should all be recorded in history."""
        from engine.decision import TradeSignal
        from execution.paper_executor import PaperExecutor
        from execution.monitor import PositionMonitor

        fetcher = make_mock_fetcher(current_price=100.0)
        predictor = make_mock_predictor(prob_up=0.7, prob_down=0.3, confidence=0.75)

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        executor = PaperExecutor(fetcher=fetcher, starting_balance=50000.0, db_path=tmp.name)

        trade_ids = []
        for i in range(3):
            signal = TradeSignal(
                pair="BTC-USDT", side="BUY", entry_price=100.0 + i,
                stop_price=98.0, target_price=104.0 + i, confidence=0.75,
                expected_move=0.02, prob_up=0.7, prob_down=0.3,
                risk_reward=2.0, atr=1.0, volatility_regime="normal",
                market_regime="uptrend", btc_correlation=0.5,
                spread_pct=0.001, volume_24h=1000000.0,
                reasoning=f"Test signal {i}",
            )
            trade_id = executor.execute_signal(signal)
            if trade_id:
                trade_ids.append(trade_id)

                # Close the trade by hitting target
                session = patch_db()
                trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
                target_price = trade.target_price
                session.close()

                fetcher_tp = make_mock_fetcher(current_price=target_price + 1.0)
                monitor = PositionMonitor(fetcher_tp, executor, predictor)
                closed = monitor.monitor_cycle()
                assert closed == trade_id

        # Verify all trades recorded
        session = patch_db()
        closed_trades = session.query(Trade).filter(Trade.status == "closed").all()
        assert len(closed_trades) == 3
        total_pnl = sum(t.pnl for t in closed_trades if t.pnl)
        assert total_pnl > 0  # All hit targets, so should be profitable
        session.close()
        os.unlink(tmp.name)


# =========================================================================
# PHASE 11: Backtester Tests
# =========================================================================

class TestBacktesterAdvanced:
    """Test backtesting engine with various scenarios."""

    def test_profitable_trade(self):
        """Backtest with a signal that should hit target."""
        from backtesting.backtester import Backtester
        np.random.seed(42)
        n = 200
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        # Create uptrending data
        close = 100 + np.linspace(0, 20, n) + np.random.randn(n) * 0.2
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": close + np.random.randn(n) * 0.05,
            "high": close + 0.5,
            "low": close - 0.2,
            "close": close,
            "volume": np.random.exponential(1000, n),
        })

        signals = [{
            "timestamp": timestamps[10],
            "pair": "TEST-USDT",
            "entry_price": float(close[10]),
            "stop_price": float(close[10] * 0.98),
            "target_price": float(close[10] * 1.04),
            "confidence": 0.75,
        }]

        bt = Backtester(initial_balance=10000)
        result = bt.run(df, signals)
        assert result.total_trades >= 1
        # With uptrending data, should hit target
        assert any(t.exit_reason == "target" for t in result.trades)

    def test_monte_carlo_gives_probability(self):
        """Monte Carlo should produce probability estimates."""
        from backtesting.backtester import Backtester, BacktestTrade
        trades = [
            BacktestTrade(
                pair="T", side="BUY",
                entry_time=datetime.now() - timedelta(hours=i),
                entry_price=100, stop_price=98, target_price=104,
                pnl=5.0 if i % 3 != 0 else -3.0,
                pnl_pct=0.05 if i % 3 != 0 else -0.03,
                r_multiple=2.0 if i % 3 != 0 else -1.5,
            )
            for i in range(30)
        ]
        bt = Backtester()
        result = bt.monte_carlo(trades, n_simulations=100)
        assert "median_final_balance" in result
        assert "probability_profit" in result
        assert 0 <= result["probability_profit"] <= 1
        assert result["median_final_balance"] > 0


# =========================================================================
# PHASE 12: Price Rounding & Position Sizing Edge Cases
# =========================================================================

class TestExecutionUtilities:
    """Test execution helper functions."""

    def test_round_price(self):
        """Price should be rounded to exchange increment."""
        from execution.paper_executor import PaperExecutor
        fetcher = make_mock_fetcher()
        # price_increment = 0.01
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        executor = PaperExecutor(fetcher=fetcher, db_path=tmp.name)
        rounded = executor.round_price(123.456789, "BTC-USDT")
        assert rounded == 123.45
        os.unlink(tmp.name)

    def test_round_quantity(self):
        """Quantity should be rounded to exchange increment."""
        from execution.paper_executor import PaperExecutor
        fetcher = make_mock_fetcher()
        # base_increment = 0.00001
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        executor = PaperExecutor(fetcher=fetcher, db_path=tmp.name)
        rounded = executor.round_quantity(1.23456789, "BTC-USDT")
        assert rounded == 1.23456
        os.unlink(tmp.name)

    def test_position_size_zero_stop_distance(self):
        """Zero stop distance should return 0 qty."""
        from execution.paper_executor import PaperExecutor
        from engine.decision import TradeSignal
        fetcher = make_mock_fetcher()
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        executor = PaperExecutor(fetcher=fetcher, db_path=tmp.name)
        signal = TradeSignal(
            pair="BTC-USDT", side="BUY", entry_price=100.0,
            stop_price=100.0,  # Same as entry = zero distance
            target_price=104.0, confidence=0.75,
            expected_move=0.02, prob_up=0.7, prob_down=0.3,
            risk_reward=2.0, atr=1.0, volatility_regime="normal",
            market_regime="uptrend", btc_correlation=0.5,
            spread_pct=0.001, volume_24h=1000000.0,
            reasoning="Test",
        )
        qty = executor.calculate_position_size(signal, 10000.0, 0.01)
        assert qty == 0.0
        os.unlink(tmp.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
