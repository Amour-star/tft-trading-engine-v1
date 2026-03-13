from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.strategy_generator.types import (
    BacktestTrade,
    PerformanceMetrics,
    StrategyCandidate,
    StrategyEvaluation,
    WalkForwardFold,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


@dataclass
class StrategyResearchBacktester:
    initial_balance: float = 10_000.0
    fee_rate: float = 0.001
    slippage_bps: float = 2.0
    min_sharpe: float = 1.5
    max_drawdown: float = 0.20
    min_profit_factor: float = 1.3

    def evaluate_candidate(
        self,
        candidate: StrategyCandidate,
        frame: pd.DataFrame,
        train_fraction: float = 0.65,
        test_fraction: float = 0.20,
        max_folds: int = 3,
    ) -> StrategyEvaluation:
        total_rows = len(frame)
        train_size = max(100, int(total_rows * train_fraction))
        test_size = max(50, int(total_rows * test_fraction))
        if train_size + test_size > total_rows:
            train_size = max(50, total_rows - test_size)
        start = 0
        folds: List[WalkForwardFold] = []

        while start + train_size + test_size <= total_rows and len(folds) < max_folds:
            train_df = frame.iloc[start : start + train_size].reset_index(drop=True)
            test_df = frame.iloc[start + train_size : start + train_size + test_size].reset_index(drop=True)
            train_metrics = self.backtest(candidate, train_df)
            test_metrics = self.backtest(candidate, test_df)
            folds.append(
                WalkForwardFold(
                    fold_index=len(folds),
                    train_start=str(train_df["timestamp"].iloc[0]),
                    train_end=str(train_df["timestamp"].iloc[-1]),
                    test_start=str(test_df["timestamp"].iloc[0]),
                    test_end=str(test_df["timestamp"].iloc[-1]),
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                )
            )
            start += test_size

        if not folds:
            whole = self.backtest(candidate, frame)
            return StrategyEvaluation(candidate=candidate, train_metrics=whole, test_metrics=whole, walk_forward_folds=[])

        train_metrics = self._aggregate([fold.train_metrics for fold in folds])
        test_metrics = self._aggregate([fold.test_metrics for fold in folds])
        return StrategyEvaluation(
            candidate=candidate,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            walk_forward_folds=folds,
        )

    def backtest(self, candidate: StrategyCandidate, frame: pd.DataFrame) -> PerformanceMetrics:
        if frame.empty or len(frame) < 30:
            return PerformanceMetrics(failure_reason="insufficient_data")

        balance = float(self.initial_balance)
        equity_curve = [balance]
        trades: List[BacktestTrade] = []
        open_position: Optional[Dict[str, Any]] = None

        for idx in range(len(frame) - 1):
            row = frame.iloc[idx]
            next_row = frame.iloc[idx + 1]

            if open_position is not None:
                exit_payload = self._check_exit(open_position, row, candidate, idx)
                if exit_payload is not None:
                    trade = self._close_trade(candidate, open_position, exit_payload, balance)
                    balance += trade.pnl
                    trades.append(trade)
                    equity_curve.append(balance)
                    open_position = None

            if open_position is not None:
                continue

            direction, confidence, reason = self._signal(candidate, row)
            if direction == 0:
                continue

            entry_price, spread_cost, slippage_cost = self._entry_fill(next_row, direction)
            atr = max(_safe_float(row.get("atr_14"), 0.0), entry_price * 0.0015)
            stop_distance = atr * candidate.stop_atr_multiplier
            take_distance = atr * candidate.take_atr_multiplier
            if stop_distance <= 0 or take_distance <= 0:
                continue

            risk_amount = balance * candidate.risk_per_trade
            quantity = risk_amount / max(stop_distance, 1e-9)
            entry_fee = quantity * entry_price * self.fee_rate
            if direction > 0 and quantity * entry_price + entry_fee > balance:
                quantity = max(0.0, (balance * 0.95) / max(entry_price * (1.0 + self.fee_rate), 1e-9))
            if quantity <= 0:
                continue

            stop_price = entry_price - stop_distance if direction > 0 else entry_price + stop_distance
            target_price = entry_price + take_distance if direction > 0 else entry_price - take_distance
            open_position = {
                "side": "BUY" if direction > 0 else "SELL",
                "direction": direction,
                "entry_time": pd.to_datetime(next_row["timestamp"]).to_pydatetime(),
                "entry_index": idx + 1,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_price": stop_price,
                "target_price": target_price,
                "trailing_stop": stop_price,
                "entry_fee": entry_fee,
                "spread_cost": spread_cost,
                "slippage_cost": slippage_cost,
                "entry_reason": reason,
                "confidence": confidence,
            }

        if open_position is not None:
            final_row = frame.iloc[-1]
            exit_payload = {
                "time": pd.to_datetime(final_row["timestamp"]).to_pydatetime(),
                "price": _safe_float(final_row.get("close"), open_position["entry_price"]),
                "reason": "end_of_data",
                "spread_cost": 0.0,
                "slippage_cost": 0.0,
                "exit_index": len(frame) - 1,
            }
            trade = self._close_trade(candidate, open_position, exit_payload, balance)
            balance += trade.pnl
            trades.append(trade)
            equity_curve.append(balance)

        metrics = self._metrics_from_trades(trades, equity_curve)
        metrics.passed = (
            metrics.total_trades > 0
            and metrics.sharpe_ratio >= self.min_sharpe
            and metrics.max_drawdown <= self.max_drawdown
            and metrics.profit_factor >= self.min_profit_factor
        )
        if not metrics.passed and not metrics.failure_reason:
            metrics.failure_reason = (
                f"sharpe={metrics.sharpe_ratio:.2f} dd={metrics.max_drawdown:.2%} "
                f"pf={metrics.profit_factor:.2f}"
            )
        return metrics

    def _signal(self, candidate: StrategyCandidate, row: pd.Series) -> Tuple[int, float, str]:
        if not self._passes_filters(candidate, row):
            return 0, 0.0, "filters_blocked"

        long_hits = 0
        short_hits = 0
        reasons: List[str] = []
        for indicator, params in candidate.entry_logic.items():
            if indicator == "rsi":
                rsi_value = _safe_float(row.get("rsi_14"), 50.0)
                if rsi_value <= _safe_float(params.get("long_below"), 30.0):
                    long_hits += 1
                    reasons.append(f"rsi_long={rsi_value:.1f}")
                if rsi_value >= _safe_float(params.get("short_above"), 70.0):
                    short_hits += 1
                    reasons.append(f"rsi_short={rsi_value:.1f}")
            elif indicator == "ema":
                gap = _safe_float(row.get("ema_gap"), 0.0)
                threshold = _safe_float(params.get("gap_threshold"), 0.001)
                if gap >= threshold:
                    long_hits += 1
                    reasons.append(f"ema_long={gap:.4f}")
                if gap <= -threshold:
                    short_hits += 1
                    reasons.append(f"ema_short={gap:.4f}")
            elif indicator == "vwap":
                deviation = _safe_float(row.get("vwap_deviation"), 0.0)
                threshold = _safe_float(params.get("threshold"), 0.002)
                mode = str(params.get("mode", "mean_reversion"))
                if mode == "trend":
                    if deviation >= threshold:
                        long_hits += 1
                        reasons.append(f"vwap_trend_long={deviation:.4f}")
                    if deviation <= -threshold:
                        short_hits += 1
                        reasons.append(f"vwap_trend_short={deviation:.4f}")
                else:
                    if deviation <= -threshold:
                        long_hits += 1
                        reasons.append(f"vwap_revert_long={deviation:.4f}")
                    if deviation >= threshold:
                        short_hits += 1
                        reasons.append(f"vwap_revert_short={deviation:.4f}")
            elif indicator == "momentum":
                window = int(params.get("window", 5))
                value = _safe_float(row.get(f"momentum_{window}"), 0.0)
                threshold = _safe_float(params.get("threshold"), 0.003)
                mode = str(params.get("mode", "trend"))
                if mode == "trend":
                    if value >= threshold:
                        long_hits += 1
                        reasons.append(f"mom_trend_long={value:.4f}")
                    if value <= -threshold:
                        short_hits += 1
                        reasons.append(f"mom_trend_short={value:.4f}")
                else:
                    if value <= -threshold:
                        long_hits += 1
                        reasons.append(f"mom_revert_long={value:.4f}")
                    if value >= threshold:
                        short_hits += 1
                        reasons.append(f"mom_revert_short={value:.4f}")
            elif indicator == "orderbook_imbalance":
                value = _safe_float(row.get("orderbook_imbalance"), 0.0)
                threshold = _safe_float(params.get("threshold"), 0.10)
                if value >= threshold:
                    long_hits += 1
                    reasons.append(f"obi_long={value:.3f}")
                if value <= -threshold:
                    short_hits += 1
                    reasons.append(f"obi_short={value:.3f}")

        confirmations = max(candidate.min_confirmations, 1)
        total_checks = max(1, len(candidate.entry_logic))
        if long_hits >= confirmations and long_hits > short_hits:
            return 1, min(0.99, long_hits / total_checks), ",".join(reasons)
        if candidate.allow_short and short_hits >= confirmations and short_hits > long_hits:
            return -1, min(0.99, short_hits / total_checks), ",".join(reasons)
        return 0, 0.0, "no_consensus"

    def _passes_filters(self, candidate: StrategyCandidate, row: pd.Series) -> bool:
        for indicator, params in candidate.filter_logic.items():
            if indicator == "volume_spike":
                if _safe_float(row.get("volume_spike"), 0.0) < _safe_float(params.get("minimum"), 1.0):
                    return False
            elif indicator == "volatility":
                value = _safe_float(row.get("rolling_volatility"), 0.0)
                threshold = _safe_float(params.get("threshold"), 0.01)
                mode = str(params.get("mode", "breakout"))
                if mode == "breakout" and value < threshold:
                    return False
                if mode == "quiet" and value > threshold:
                    return False
            elif indicator == "atr":
                atr_pct = _safe_float(row.get("atr_pct"), 0.0)
                if atr_pct < _safe_float(params.get("min_atr_pct"), 0.0):
                    return False
                if atr_pct > _safe_float(params.get("max_atr_pct"), 1.0):
                    return False
        return True

    def _entry_fill(self, next_row: pd.Series, direction: int) -> Tuple[float, float, float]:
        open_price = _safe_float(next_row.get("open"), next_row.get("close"))
        spread = max(_safe_float(next_row.get("spread_pct"), 0.0005), 0.0)
        slippage = self.slippage_bps / 10_000.0
        price_multiplier = 1.0 + spread * 0.5 + slippage
        if direction < 0:
            price_multiplier = 1.0 - spread * 0.5 - slippage
        entry_price = open_price * price_multiplier
        return entry_price, open_price * spread * 0.5, open_price * slippage

    def _check_exit(
        self,
        position: Dict[str, Any],
        row: pd.Series,
        candidate: StrategyCandidate,
        current_index: int,
    ) -> Optional[Dict[str, Any]]:
        direction = int(position["direction"])
        high = _safe_float(row.get("high"), row.get("close"))
        low = _safe_float(row.get("low"), row.get("close"))
        close = _safe_float(row.get("close"), position["entry_price"])
        atr = max(_safe_float(row.get("atr_14"), 0.0), close * 0.0010)
        trail_gap = atr * candidate.trailing_atr_multiplier

        if direction > 0:
            position["trailing_stop"] = max(position["trailing_stop"], close - trail_gap)
            stop_level = max(position["stop_price"], position["trailing_stop"])
            if low <= stop_level:
                return self._exit_payload(row, stop_level, "stop_or_trail", current_index)
            if high >= position["target_price"]:
                return self._exit_payload(row, position["target_price"], "target", current_index)
        else:
            position["trailing_stop"] = min(position["trailing_stop"], close + trail_gap)
            stop_level = min(position["stop_price"], position["trailing_stop"])
            if high >= stop_level:
                return self._exit_payload(row, stop_level, "stop_or_trail", current_index)
            if low <= position["target_price"]:
                return self._exit_payload(row, position["target_price"], "target", current_index)

        if current_index - int(position["entry_index"]) >= candidate.max_hold_bars:
            return self._exit_payload(row, close, "time_stop", current_index)
        return None

    def _exit_payload(self, row: pd.Series, price: float, reason: str, exit_index: int) -> Dict[str, Any]:
        return {
            "time": pd.to_datetime(row["timestamp"]).to_pydatetime(),
            "price": float(price),
            "reason": reason,
            "spread_cost": _safe_float(row.get("spread_pct"), 0.0) * float(price) * 0.5,
            "slippage_cost": float(price) * self.slippage_bps / 10_000.0,
            "exit_index": exit_index,
        }

    def _close_trade(
        self,
        candidate: StrategyCandidate,
        position: Dict[str, Any],
        exit_payload: Dict[str, Any],
        balance_before: float,
    ) -> BacktestTrade:
        exit_price = _safe_float(exit_payload["price"], position["entry_price"])
        quantity = _safe_float(position["quantity"], 0.0)
        direction = int(position["direction"])
        gross_pnl = (exit_price - position["entry_price"]) * quantity * direction
        exit_fee = exit_price * quantity * self.fee_rate
        total_fees = _safe_float(position["entry_fee"], 0.0) + exit_fee
        net_pnl = gross_pnl - total_fees
        notional = max(position["entry_price"] * quantity, 1e-9)
        return_pct = net_pnl / max(balance_before, 1e-9)
        pnl_pct = net_pnl / notional
        hold_bars = int(exit_payload["exit_index"]) - int(position["entry_index"])
        return BacktestTrade(
            strategy_id=candidate.strategy_id,
            symbol=candidate.symbol,
            side=str(position["side"]),
            entry_time=position["entry_time"],
            exit_time=exit_payload["time"],
            entry_price=_safe_float(position["entry_price"]),
            exit_price=exit_price,
            quantity=quantity,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            return_pct=return_pct,
            hold_bars=max(1, hold_bars),
            exit_reason=str(exit_payload["reason"]),
            fees_paid=total_fees,
            spread_cost=_safe_float(position["spread_cost"], 0.0) + _safe_float(exit_payload["spread_cost"], 0.0),
            slippage_cost=_safe_float(position["slippage_cost"], 0.0) + _safe_float(exit_payload["slippage_cost"], 0.0),
        )

    def _metrics_from_trades(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[float],
    ) -> PerformanceMetrics:
        if not trades:
            return PerformanceMetrics(
                total_trades=0,
                failure_reason="no_trades",
                equity_curve=equity_curve,
            )

        returns = np.asarray([trade.return_pct for trade in trades], dtype=np.float64)
        pnls = np.asarray([trade.pnl for trade in trades], dtype=np.float64)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        downside = returns[returns < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        sharpe = (mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 1e-9 else 0.0
        sortino = (mean_ret / downside_std * np.sqrt(252.0)) if downside_std > 1e-9 else 0.0

        equity = np.asarray(equity_curve, dtype=np.float64)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / np.where(peaks == 0.0, 1.0, peaks)
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0
        win_rate = float(np.mean(pnls > 0))
        gross_profit = float(np.sum(wins)) if len(wins) else 0.0
        gross_loss = abs(float(np.sum(losses))) if len(losses) else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else gross_profit
        total_return = (equity_curve[-1] - equity_curve[0]) / max(equity_curve[0], 1e-9)
        expectancy = float(np.mean(pnls))
        avg_trade_return = float(np.mean(returns))
        score = (
            sharpe * 0.40
            + sortino * 0.20
            + profit_factor * 0.20
            + total_return * 12.0
            + win_rate * 0.15
            - max_drawdown * 1.5
        )
        return PerformanceMetrics(
            total_trades=len(trades),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return=total_return,
            avg_trade_return=avg_trade_return,
            expectancy=expectancy,
            score=score,
            equity_curve=list(map(float, equity_curve)),
            trade_returns=list(map(float, returns.tolist())),
            trades=trades,
        )

    def _aggregate(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        valid = [metrics for metrics in metrics_list if metrics.total_trades > 0]
        if not valid:
            return PerformanceMetrics(failure_reason="no_valid_folds")

        total_trades = sum(item.total_trades for item in valid)
        weights = [item.total_trades / total_trades for item in valid]
        aggregate = PerformanceMetrics(
            total_trades=total_trades,
            sharpe_ratio=sum(item.sharpe_ratio * w for item, w in zip(valid, weights)),
            sortino_ratio=sum(item.sortino_ratio * w for item, w in zip(valid, weights)),
            max_drawdown=max(item.max_drawdown for item in valid),
            win_rate=sum(item.win_rate * w for item, w in zip(valid, weights)),
            profit_factor=sum(item.profit_factor * w for item, w in zip(valid, weights)),
            total_return=sum(item.total_return * w for item, w in zip(valid, weights)),
            avg_trade_return=sum(item.avg_trade_return * w for item, w in zip(valid, weights)),
            expectancy=sum(item.expectancy * w for item, w in zip(valid, weights)),
            score=sum(item.score * w for item, w in zip(valid, weights)),
            failure_reason="",
        )
        aggregate.passed = (
            aggregate.sharpe_ratio >= self.min_sharpe
            and aggregate.max_drawdown <= self.max_drawdown
            and aggregate.profit_factor >= self.min_profit_factor
        )
        if not aggregate.passed:
            aggregate.failure_reason = (
                f"oos_sharpe={aggregate.sharpe_ratio:.2f} "
                f"oos_dd={aggregate.max_drawdown:.2%} "
                f"oos_pf={aggregate.profit_factor:.2f}"
            )
        return aggregate
