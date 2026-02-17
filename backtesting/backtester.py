"""
Backtesting engine with walk-forward validation and Monte Carlo simulation.
Realistic simulation including slippage, spread, latency, and commissions.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings


@dataclass
class BacktestTrade:
    pair: str
    side: str
    entry_time: datetime
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    confidence: float = 0.0


@dataclass
class BacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_r_multiple: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)


class Backtester:
    """
    Walk-forward backtester with realistic cost modeling.
    """

    def __init__(
        self,
        slippage_bps: Optional[float] = None,
        commission_bps: Optional[float] = None,
        latency_ms: Optional[int] = None,
        initial_balance: float = 10000.0,
    ) -> None:
        cfg = settings.backtest
        self.slippage_bps = slippage_bps or cfg.slippage_bps
        self.commission_bps = commission_bps or cfg.commission_bps
        self.latency_ms = latency_ms or cfg.latency_ms
        self.initial_balance = initial_balance

    def run(
        self,
        df: pd.DataFrame,
        signals: List[Dict[str, Any]],
        risk_per_trade: float = 0.01,
    ) -> BacktestResult:
        """
        Run backtest on historical data with generated signals.

        Parameters
        ----------
        df : pd.DataFrame
            Full OHLCV DataFrame with timestamp index
        signals : list of dict
            Each signal: {timestamp, pair, side, entry_price, stop_price,
                         target_price, confidence}
        risk_per_trade : float
            Risk percentage per trade
        """
        result = BacktestResult()
        balance = self.initial_balance
        equity_curve = [balance]
        peak_balance = balance

        for signal in signals:
            if balance <= 0:
                break

            trade = self._simulate_trade(df, signal, balance, risk_per_trade)
            if trade is None:
                continue

            balance += trade.pnl
            equity_curve.append(balance)
            result.trades.append(trade)

            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            drawdown = peak_balance - balance
            drawdown_pct = drawdown / peak_balance if peak_balance > 0 else 0
            if drawdown > result.max_drawdown:
                result.max_drawdown = drawdown
                result.max_drawdown_pct = drawdown_pct

        # Compute statistics
        result.total_trades = len(result.trades)
        result.winning_trades = sum(1 for t in result.trades if t.pnl > 0)
        result.losing_trades = result.total_trades - result.winning_trades
        result.total_pnl = balance - self.initial_balance
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        result.equity_curve = equity_curve

        r_multiples = [t.r_multiple for t in result.trades if t.r_multiple != 0]
        result.avg_r_multiple = np.mean(r_multiples) if r_multiples else 0

        wins = [t.pnl for t in result.trades if t.pnl > 0]
        losses = [t.pnl for t in result.trades if t.pnl < 0]
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0
        result.profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0

        # Sharpe ratio (annualized, assuming ~250 trading days)
        if result.trades:
            returns = pd.Series([t.pnl_pct for t in result.trades])
            if returns.std() > 0:
                daily_returns = returns.mean()
                daily_std = returns.std()
                result.sharpe_ratio = (daily_returns / daily_std) * np.sqrt(250)
            else:
                result.sharpe_ratio = 0
        else:
            result.sharpe_ratio = 0

        # Pass/fail check
        result.passed = True
        if result.sharpe_ratio < settings.backtest.min_sharpe:
            result.passed = False
            result.failure_reasons.append(
                f"Sharpe {result.sharpe_ratio:.2f} < {settings.backtest.min_sharpe}"
            )
        if result.max_drawdown_pct > settings.backtest.max_drawdown_pct:
            result.passed = False
            result.failure_reasons.append(
                f"Max DD {result.max_drawdown_pct:.2%} > {settings.backtest.max_drawdown_pct:.2%}"
            )

        logger.info(
            f"Backtest: {result.total_trades} trades | "
            f"Win rate: {result.win_rate:.2%} | "
            f"Sharpe: {result.sharpe_ratio:.2f} | "
            f"Max DD: {result.max_drawdown_pct:.2%} | "
            f"PnL: {result.total_pnl:.2f} | "
            f"Passed: {result.passed}"
        )

        return result

    def _simulate_trade(
        self,
        df: pd.DataFrame,
        signal: Dict[str, Any],
        balance: float,
        risk_pct: float,
    ) -> Optional[BacktestTrade]:
        """Simulate a single trade with realistic costs."""
        entry_time = signal["timestamp"]
        entry_price = signal["entry_price"]
        stop_price = signal["stop_price"]
        target_price = signal["target_price"]

        # Apply slippage to entry
        slippage = entry_price * (self.slippage_bps / 10000)
        entry_price += slippage  # Worse fill for buy

        # Commission on entry
        commission_rate = self.commission_bps / 10000

        # Position sizing
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return None
        risk_amount = balance * risk_pct
        quantity = risk_amount / stop_distance
        cost = quantity * entry_price * (1 + commission_rate)
        if cost > balance * 0.95:
            quantity = (balance * 0.90) / (entry_price * (1 + commission_rate))

        # Simulate price action after entry
        mask = df["timestamp"] > entry_time
        future_df = df[mask]
        if future_df.empty:
            return None

        exit_price = None
        exit_time = None
        exit_reason = ""

        for _, row in future_df.iterrows():
            # Check stop hit (using low)
            if row["low"] <= stop_price:
                exit_price = stop_price - slippage  # Slippage on exit too
                exit_time = row["timestamp"]
                exit_reason = "stop"
                break
            # Check target hit (using high)
            if row["high"] >= target_price:
                exit_price = target_price - slippage * 0.5  # Less slippage on limit
                exit_time = row["timestamp"]
                exit_reason = "target"
                break

        if exit_price is None:
            # Trade still open at end of data - close at last price
            exit_price = float(future_df.iloc[-1]["close"])
            exit_time = future_df.iloc[-1]["timestamp"]
            exit_reason = "end_of_data"

        # Calculate PnL
        gross_pnl = (exit_price - entry_price) * quantity
        commission_total = (entry_price + exit_price) * quantity * commission_rate
        net_pnl = gross_pnl - commission_total
        pnl_pct = (exit_price - entry_price) / entry_price
        r_multiple = (exit_price - entry_price) / stop_distance if stop_distance > 0 else 0

        return BacktestTrade(
            pair=signal.get("pair", "UNKNOWN"),
            side="BUY",
            entry_time=entry_time,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            confidence=signal.get("confidence", 0),
        )

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        df: pd.DataFrame,
        signal_generator,  # callable(df_window) -> List[signal]
        window_size: int = 2000,
        step_size: int = 500,
        risk_per_trade: float = 0.01,
    ) -> BacktestResult:
        """
        Walk-forward backtest.
        Trains on window, tests on next step, rolls forward.
        """
        all_trades: List[BacktestTrade] = []
        n = len(df)

        for start in range(0, n - window_size, step_size):
            train_end = start + window_size
            test_end = min(train_end + step_size, n)

            train_df = df.iloc[start:train_end]
            test_df = df.iloc[train_end:test_end]

            if test_df.empty:
                break

            # Generate signals on training data patterns, apply to test
            signals = signal_generator(train_df, test_df)
            if not signals:
                continue

            result = self.run(test_df, signals, risk_per_trade)
            all_trades.extend(result.trades)

        # Aggregate
        return self._aggregate_results(all_trades)

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def monte_carlo(
        self,
        trades: List[BacktestTrade],
        n_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation for robustness testing.
        Randomly reorders trades to test if results are dependent on sequence.
        """
        if not trades:
            return {"passed": False, "reason": "No trades"}

        pnls = [t.pnl for t in trades]
        initial = self.initial_balance

        final_balances = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(n_simulations):
            shuffled = random.sample(pnls, len(pnls))
            balance = initial
            peak = initial
            max_dd = 0

            returns_list = []
            for pnl in shuffled:
                balance += pnl
                returns_list.append(pnl / peak if peak > 0 else 0)
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            final_balances.append(balance)
            max_drawdowns.append(max_dd)

            ret_series = pd.Series(returns_list)
            if ret_series.std() > 0:
                sharpe = (ret_series.mean() / ret_series.std()) * np.sqrt(250)
            else:
                sharpe = 0
            sharpe_ratios.append(sharpe)

        result = {
            "n_simulations": n_simulations,
            "median_final_balance": float(np.median(final_balances)),
            "p5_final_balance": float(np.percentile(final_balances, 5)),
            "p95_final_balance": float(np.percentile(final_balances, 95)),
            "median_max_drawdown": float(np.median(max_drawdowns)),
            "p95_max_drawdown": float(np.percentile(max_drawdowns, 95)),
            "median_sharpe": float(np.median(sharpe_ratios)),
            "p5_sharpe": float(np.percentile(sharpe_ratios, 5)),
            "probability_profit": float(np.mean([b > initial for b in final_balances])),
            "probability_ruin": float(np.mean([b < initial * 0.5 for b in final_balances])),
        }

        # Pass if 5th percentile Sharpe still meets threshold
        result["passed"] = (
            result["p5_sharpe"] > settings.backtest.min_sharpe * 0.7
            and result["p95_max_drawdown"] < settings.backtest.max_drawdown_pct * 1.5
            and result["probability_profit"] > 0.6
        )

        logger.info(
            f"Monte Carlo ({n_simulations} sims): "
            f"Median balance: {result['median_final_balance']:.2f} | "
            f"P5 Sharpe: {result['p5_sharpe']:.2f} | "
            f"P95 DD: {result['p95_max_drawdown']:.2%} | "
            f"Prob profit: {result['probability_profit']:.2%} | "
            f"Passed: {result['passed']}"
        )

        return result

    def _aggregate_results(self, trades: List[BacktestTrade]) -> BacktestResult:
        """Aggregate trades into a single BacktestResult."""
        result = BacktestResult()
        result.trades = trades
        result.total_trades = len(trades)
        result.winning_trades = sum(1 for t in trades if t.pnl > 0)
        result.losing_trades = result.total_trades - result.winning_trades
        result.total_pnl = sum(t.pnl for t in trades)
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0

        balance = self.initial_balance
        peak = balance
        equity = [balance]
        max_dd = 0
        max_dd_pct = 0

        for t in trades:
            balance += t.pnl
            equity.append(balance)
            if balance > peak:
                peak = balance
            dd = peak - balance
            dd_pct = dd / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        result.equity_curve = equity
        result.max_drawdown = max_dd
        result.max_drawdown_pct = max_dd_pct

        r_mults = [t.r_multiple for t in trades]
        result.avg_r_multiple = float(np.mean(r_mults)) if r_mults else 0

        returns = pd.Series([t.pnl_pct for t in trades])
        if len(returns) > 1 and returns.std() > 0:
            result.sharpe_ratio = float((returns.mean() / returns.std()) * np.sqrt(250))
        else:
            result.sharpe_ratio = 0

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        result.avg_win = float(np.mean(wins)) if wins else 0
        result.avg_loss = float(np.mean(losses)) if losses else 0
        result.profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0

        result.passed = (
            result.sharpe_ratio >= settings.backtest.min_sharpe
            and result.max_drawdown_pct <= settings.backtest.max_drawdown_pct
        )
        if not result.passed:
            if result.sharpe_ratio < settings.backtest.min_sharpe:
                result.failure_reasons.append(f"Sharpe {result.sharpe_ratio:.2f}")
            if result.max_drawdown_pct > settings.backtest.max_drawdown_pct:
                result.failure_reasons.append(f"Max DD {result.max_drawdown_pct:.2%}")

        return result
