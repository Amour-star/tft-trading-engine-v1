"""
Institutional-grade performance metrics module.
Tracks per-symbol and portfolio-level trading performance.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from config.settings import ACTIVE_SYMBOL, settings
from data.database import Trade, get_session


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


class PerformanceTracker:
    """Computes institutional performance metrics per symbol."""

    def __init__(self, symbol: str = ACTIVE_SYMBOL) -> None:
        self.symbol = symbol

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all performance metrics for this symbol."""
        session = get_session()
        try:
            trades = (
                session.query(Trade)
                .filter(Trade.status == "closed", Trade.pair == self.symbol, Trade.pnl.isnot(None))
                .order_by(Trade.exit_time.asc())
                .all()
            )

            if not trades:
                return self._empty_metrics()

            pnls = [_safe_float(t.pnl) for t in trades]
            pnl_pcts = self._realized_return_series(pnls)
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            total_trades = len(trades)
            win_count = len(wins)
            loss_count = len(losses)
            win_rate = win_count / total_trades if total_trades > 0 else 0.0
            avg_win = float(np.mean(wins)) if wins else 0.0
            avg_loss = float(np.mean(losses)) if losses else 0.0

            # Expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

            # Profit Factor = gross_wins / abs(gross_losses)
            gross_wins = sum(wins)
            gross_losses = abs(sum(losses))
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else max(gross_wins, 0.0)

            min_trades_for_ratios = max(2, int(getattr(settings.trading, "metrics_min_trades", 20)))
            mean_ret = float(np.mean(pnl_pcts)) if pnl_pcts else 0.0
            std_ret = float(np.std(pnl_pcts, ddof=1)) if len(pnl_pcts) >= 2 else 0.0
            sharpe = 0.0
            sortino = 0.0
            if total_trades >= min_trades_for_ratios and len(pnl_pcts) >= 2 and std_ret > 1e-12:
                sharpe = mean_ret / std_ret * math.sqrt(96)
                negative_returns = [r for r in pnl_pcts if r < 0]
                downside_std = float(np.std(negative_returns, ddof=1)) if len(negative_returns) >= 2 else 0.0
                sortino = (mean_ret / downside_std * math.sqrt(96)) if downside_std > 1e-12 else 0.0

            rolling_returns = pnl_pcts[-max(20, min_trades_for_ratios):]
            rolling_volatility = (
                float(np.std(rolling_returns, ddof=1))
                if len(rolling_returns) >= 2
                else 0.0
            )

            # Max Drawdown
            max_dd = self._compute_max_drawdown(pnls)

            # Equity Curve
            equity_curve = self._compute_equity_curve(pnls)

            # Exposure %
            open_trades = (
                session.query(Trade)
                .filter(Trade.status == "open", Trade.pair == self.symbol)
                .count()
            )
            exposure_pct = (open_trades / max(1, settings.trading.max_open_trades)) * 100

            # Trade Frequency per hour
            if len(trades) >= 2:
                first_trade = trades[0].exit_time or trades[0].entry_time
                last_trade = trades[-1].exit_time or trades[-1].entry_time
                hours_span = max(1, (last_trade - first_trade).total_seconds() / 3600)
                trade_freq_per_hour = total_trades / hours_span
            else:
                trade_freq_per_hour = 0.0

            window_1h = self._window_metrics(trades, timedelta(hours=1))
            window_24h = self._window_metrics(trades, timedelta(hours=24))

            lifetime = {
                "trades": int(total_trades),
                "win_rate": round(win_rate, 4),
                "total_pnl": round(sum(pnls), 4),
                "average_trade": round(float(np.mean(pnls)) if pnls else 0.0, 4),
                "sharpe": round(sharpe, 4),
                "sortino": round(sortino, 4),
                "max_drawdown": round(max_dd, 4),
            }

            metrics = {
                "symbol": self.symbol,
                "total_trades": total_trades,
                "win_rate": round(win_rate, 4),
                "avg_win": round(avg_win, 4),
                "avg_loss": round(avg_loss, 4),
                "average_trade": round(float(np.mean(pnls)) if pnls else 0.0, 4),
                "expectancy": round(expectancy, 4),
                "profit_factor": round(profit_factor, 4),
                "sharpe_ratio": round(sharpe, 4),
                "sortino_ratio": round(sortino, 4),
                "max_drawdown": round(max_dd, 4),
                "equity_curve": equity_curve[-20:],  # Last 20 points
                "exposure_pct": round(exposure_pct, 2),
                "rolling_volatility": round(rolling_volatility, 6),
                "trade_frequency_per_hour": round(trade_freq_per_hour, 4),
                "total_pnl": round(sum(pnls), 4),
                "win_count": win_count,
                "loss_count": loss_count,
                "lifetime": lifetime,
                "last_1h": window_1h,
                "last_24h": window_24h,
                "timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                f"METRICS | symbol={self.symbol} | trades={total_trades} | "
                f"win_rate={win_rate:.2%} | sharpe={sharpe:.2f} | "
                f"max_dd={max_dd:.2%} | expectancy={expectancy:.4f} | "
                f"pf={profit_factor:.2f}"
            )

            return metrics

        except Exception as exc:
            logger.error(f"Error computing metrics for {self.symbol}: {exc}")
            return self._empty_metrics()
        finally:
            session.close()

    def _compute_max_drawdown(self, pnls: List[float]) -> float:
        equity = settings.trading.paper_starting_balance
        peak = equity
        max_dd = 0.0
        for pnl in pnls:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def _compute_equity_curve(self, pnls: List[float]) -> List[float]:
        equity = settings.trading.paper_starting_balance
        curve = [equity]
        for pnl in pnls:
            equity += pnl
            curve.append(round(equity, 2))
        return curve

    def _realized_return_series(self, pnls: List[float]) -> List[float]:
        returns: List[float] = []
        equity = max(_safe_float(settings.trading.paper_starting_balance, 1.0), 1e-9)
        for pnl in pnls:
            ret = pnl / equity if abs(equity) > 1e-9 else 0.0
            returns.append(_safe_float(ret, 0.0))
            equity += pnl
            if not math.isfinite(equity) or abs(equity) <= 1e-9:
                equity = max(_safe_float(settings.trading.paper_starting_balance, 1.0), 1e-9)
        return returns

    def _window_metrics(self, trades: List[Trade], window: timedelta) -> Dict[str, float]:
        now = datetime.utcnow()
        window_rows = [
            trade
            for trade in trades
            if (trade.exit_time or trade.entry_time) and (trade.exit_time or trade.entry_time) >= (now - window)
        ]
        if not window_rows:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_trade": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
            }

        pnls = [_safe_float(trade.pnl, 0.0) for trade in window_rows]
        wins = sum(1 for pnl in pnls if pnl > 0)
        trade_count = len(pnls)
        win_rate = wins / trade_count if trade_count > 0 else 0.0
        returns = self._realized_return_series(pnls)
        min_window_trades = max(2, int(getattr(settings.trading, "metrics_min_window_trades", 5)))
        sharpe = 0.0
        sortino = 0.0
        if trade_count >= min_window_trades and len(returns) >= 2:
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns, ddof=1))
            if std_ret > 1e-12:
                sharpe = mean_ret / std_ret * math.sqrt(96)
            negatives = [r for r in returns if r < 0]
            downside = float(np.std(negatives, ddof=1)) if len(negatives) >= 2 else 0.0
            if downside > 1e-12:
                sortino = mean_ret / downside * math.sqrt(96)

        return {
            "trades": int(trade_count),
            "win_rate": round(win_rate, 4),
            "total_pnl": round(sum(pnls), 4),
            "average_trade": round(float(np.mean(pnls)) if pnls else 0.0, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "average_trade": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "equity_curve": [settings.trading.paper_starting_balance],
            "exposure_pct": 0.0,
            "rolling_volatility": 0.0,
            "trade_frequency_per_hour": 0.0,
            "total_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "lifetime": {
                "trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_trade": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
            },
            "last_1h": {
                "trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_trade": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
            },
            "last_24h": {
                "trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_trade": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }


class PortfolioMetrics:
    """Portfolio-level metrics across all symbols."""

    def __init__(self, symbols: Optional[List[str]] = None) -> None:
        self.symbols = symbols or [ACTIVE_SYMBOL]

    def compute_portfolio_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics across all symbols."""
        per_symbol = {}

        for symbol in self.symbols:
            tracker = PerformanceTracker(symbol)
            per_symbol[symbol] = tracker.compute_metrics()

        session = get_session()
        try:
            # Get all closed trades for portfolio-level stats
            all_trades = (
                session.query(Trade)
                .filter(Trade.status == "closed", Trade.pnl.isnot(None))
                .order_by(Trade.exit_time.asc())
                .all()
            )

            if not all_trades:
                return {
                    "per_symbol": per_symbol,
                    "portfolio_sharpe": 0.0,
                    "portfolio_sortino": 0.0,
                    "portfolio_win_rate": 0.0,
                    "rolling_volatility": 0.0,
                    "risk_adjusted_return": 0.0,
                    "total_portfolio_pnl": 0.0,
                    "last_1h": {"trades": 0, "total_pnl": 0.0, "win_rate": 0.0},
                    "last_24h": {"trades": 0, "total_pnl": 0.0, "win_rate": 0.0},
                    "timestamp": datetime.utcnow().isoformat(),
                }

            portfolio_pnls = [_safe_float(t.pnl) for t in all_trades]
            portfolio_pnl_pcts = self._portfolio_returns(portfolio_pnls)
            min_trades_for_ratios = max(2, int(getattr(settings.trading, "metrics_min_trades", 20)))
            win_rate = (
                sum(1 for pnl in portfolio_pnls if pnl > 0) / len(portfolio_pnls)
                if portfolio_pnls
                else 0.0
            )

            # Portfolio Sharpe
            if len(portfolio_pnl_pcts) >= 2 and len(portfolio_pnl_pcts) >= min_trades_for_ratios:
                mean_ret = float(np.mean(portfolio_pnl_pcts))
                std_ret = float(np.std(portfolio_pnl_pcts, ddof=1))
                portfolio_sharpe = (mean_ret / std_ret * math.sqrt(96)) if std_ret > 1e-12 else 0.0
                negatives = [r for r in portfolio_pnl_pcts if r < 0]
                downside = float(np.std(negatives, ddof=1)) if len(negatives) >= 2 else 0.0
                portfolio_sortino = (mean_ret / downside * math.sqrt(96)) if downside > 1e-12 else 0.0
            else:
                portfolio_sharpe = 0.0
                portfolio_sortino = 0.0

            # Rolling volatility (last 20 trades)
            recent_rets = portfolio_pnl_pcts[-20:]
            rolling_vol = float(np.std(recent_rets, ddof=1)) if len(recent_rets) >= 2 else 0.0

            # Risk-adjusted return
            total_return = sum(portfolio_pnl_pcts)
            risk_adj_return = total_return / rolling_vol if rolling_vol > 1e-12 else 0.0

            total_portfolio_pnl = sum(portfolio_pnls)
            now = datetime.utcnow()
            trades_1h = [
                _safe_float(t.pnl, 0.0)
                for t in all_trades
                if (t.exit_time or t.entry_time) and (t.exit_time or t.entry_time) >= (now - timedelta(hours=1))
            ]
            trades_24h = [
                _safe_float(t.pnl, 0.0)
                for t in all_trades
                if (t.exit_time or t.entry_time) and (t.exit_time or t.entry_time) >= (now - timedelta(hours=24))
            ]

            portfolio = {
                "per_symbol": per_symbol,
                "portfolio_sharpe": round(portfolio_sharpe, 4),
                "portfolio_sortino": round(portfolio_sortino, 4),
                "portfolio_win_rate": round(win_rate, 4),
                "rolling_volatility": round(rolling_vol, 6),
                "risk_adjusted_return": round(risk_adj_return, 4),
                "total_portfolio_pnl": round(total_portfolio_pnl, 4),
                "last_1h": {
                    "trades": int(len(trades_1h)),
                    "total_pnl": round(sum(trades_1h), 4),
                    "win_rate": round(
                        (sum(1 for pnl in trades_1h if pnl > 0) / len(trades_1h)) if trades_1h else 0.0,
                        4,
                    ),
                },
                "last_24h": {
                    "trades": int(len(trades_24h)),
                    "total_pnl": round(sum(trades_24h), 4),
                    "win_rate": round(
                        (sum(1 for pnl in trades_24h if pnl > 0) / len(trades_24h)) if trades_24h else 0.0,
                        4,
                    ),
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                f"PORTFOLIO METRICS | sharpe={portfolio_sharpe:.2f} | "
                f"vol={rolling_vol:.4f} | risk_adj={risk_adj_return:.2f} | "
                f"total_pnl={total_portfolio_pnl:.2f}"
            )

            return portfolio

        except Exception as exc:
            logger.error(f"Error computing portfolio metrics: {exc}")
            return {"per_symbol": per_symbol, "error": str(exc)}
        finally:
            session.close()

    @staticmethod
    def _portfolio_returns(pnls: List[float]) -> List[float]:
        returns: List[float] = []
        equity = max(_safe_float(settings.trading.paper_starting_balance, 1.0), 1e-9)
        for pnl in pnls:
            ret = pnl / equity if abs(equity) > 1e-9 else 0.0
            returns.append(_safe_float(ret, 0.0))
            equity += pnl
            if not math.isfinite(equity) or abs(equity) <= 1e-9:
                equity = max(_safe_float(settings.trading.paper_starting_balance, 1.0), 1e-9)
        return returns
