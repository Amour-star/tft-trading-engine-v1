from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np

from data.database import EquityHistory, MetricSnapshot, PerformanceMetric, Trade, get_session
from quant.types import PortfolioState


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


class PerformanceAnalyticsEngine:
    """Computes institutional metrics and persists snapshots continuously."""

    def update(self, portfolio: PortfolioState) -> Dict[str, float]:
        metrics = self._compute_metrics()
        self._persist_equity(portfolio)
        self._persist_metrics(portfolio, metrics)
        return metrics

    def _compute_metrics(self) -> Dict[str, float]:
        session = get_session()
        try:
            closed = (
                session.query(Trade)
                .filter(Trade.status == "closed", Trade.pnl.isnot(None))
                .order_by(Trade.exit_time.asc(), Trade.id.asc())
                .all()
            )
            if not closed:
                return {
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "average_trade": 0.0,
                    "volatility": 0.0,
                    "total_trades": 0.0,
                    "winning_trades": 0.0,
                    "losing_trades": 0.0,
                }

            pnls = np.asarray([_safe_float(row.pnl, 0.0) for row in closed], dtype=np.float64)
            rets = np.asarray([_safe_float(row.pnl_pct, 0.0) for row in closed], dtype=np.float64)
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            total = len(pnls)
            win_rate = float(len(wins) / total) if total else 0.0
            mean_ret = float(np.mean(rets)) if len(rets) else 0.0
            std_ret = float(np.std(rets, ddof=1)) if len(rets) >= 2 else 0.0
            sharpe = (mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 1e-9 else 0.0
            down = rets[rets < 0]
            down_std = float(np.std(down, ddof=1)) if len(down) >= 2 else 0.0
            sortino = (mean_ret / down_std * np.sqrt(252.0)) if down_std > 1e-9 else 0.0

            running = np.cumsum(pnls)
            peaks = np.maximum.accumulate(running)
            drawdowns = running - peaks
            max_dd = abs(float(np.min(drawdowns))) if len(drawdowns) else 0.0
            gross_profit = float(np.sum(wins)) if len(wins) else 0.0
            gross_loss = abs(float(np.sum(losses))) if len(losses) else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else gross_profit
            average_trade = float(np.mean(pnls))
            volatility = float(np.std(rets, ddof=1)) if len(rets) >= 2 else 0.0

            return {
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "average_trade": average_trade,
                "volatility": volatility,
                "total_trades": float(total),
                "winning_trades": float(len(wins)),
                "losing_trades": float(len(losses)),
            }
        finally:
            session.close()

    def _persist_equity(self, portfolio: PortfolioState) -> None:
        session = get_session()
        try:
            per_symbol_exposure = defaultdict(float)
            per_symbol_count = defaultdict(int)
            for position in portfolio.open_positions:
                per_symbol_exposure[position.symbol] += abs(position.quantity * position.avg_entry_price)
                per_symbol_count[position.symbol] += 1

            # Portfolio row.
            session.add(
                EquityHistory(
                    timestamp=portfolio.timestamp,
                    symbol="PORTFOLIO",
                    mode="PAPER",
                    balance=_safe_float(portfolio.balance, 0.0),
                    realized_pnl=_safe_float(portfolio.realized_pnl, 0.0),
                    unrealized_pnl=_safe_float(portfolio.unrealized_pnl, 0.0),
                    equity=_safe_float(portfolio.equity, 0.0),
                    exposure=_safe_float(portfolio.exposure_notional, 0.0),
                    open_positions=len(portfolio.open_positions),
                )
            )

            # Per-symbol rows for dashboard/API.
            for symbol, exposure in per_symbol_exposure.items():
                session.add(
                    EquityHistory(
                        timestamp=portfolio.timestamp,
                        symbol=symbol,
                        mode="PAPER",
                        balance=_safe_float(portfolio.balance, 0.0),
                        realized_pnl=_safe_float(portfolio.realized_pnl, 0.0),
                        unrealized_pnl=0.0,
                        equity=_safe_float(portfolio.equity, 0.0),
                        exposure=_safe_float(exposure, 0.0),
                        open_positions=int(per_symbol_count[symbol]),
                    )
                )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _persist_metrics(self, portfolio: PortfolioState, metrics: Dict[str, float]) -> None:
        session = get_session()
        try:
            snapshot = MetricSnapshot(
                timestamp=portfolio.timestamp,
                symbol="PORTFOLIO",
                sharpe=_safe_float(metrics.get("sharpe_ratio"), 0.0),
                sortino=_safe_float(metrics.get("sortino_ratio"), 0.0),
                max_drawdown=_safe_float(metrics.get("max_drawdown"), 0.0),
                win_rate=_safe_float(metrics.get("win_rate"), 0.0),
                profit_factor=_safe_float(metrics.get("profit_factor"), 0.0),
                average_trade=_safe_float(metrics.get("average_trade"), 0.0),
                exposure=_safe_float(portfolio.exposure_notional, 0.0),
                equity=_safe_float(portfolio.equity, 0.0),
                rolling_volatility=_safe_float(metrics.get("volatility"), 0.0),
                total_trades=int(_safe_float(metrics.get("total_trades"), 0.0)),
                winning_trades=int(_safe_float(metrics.get("winning_trades"), 0.0)),
                losing_trades=int(_safe_float(metrics.get("losing_trades"), 0.0)),
            )
            session.add(snapshot)
            session.add(
                PerformanceMetric(
                    timestamp=portfolio.timestamp.isoformat(),
                    equity=_safe_float(portfolio.equity, 0.0),
                    return_value=_safe_float(metrics.get("average_trade"), 0.0),
                    sharpe=_safe_float(metrics.get("sharpe_ratio"), 0.0),
                    sortino=_safe_float(metrics.get("sortino_ratio"), 0.0),
                    max_drawdown=_safe_float(metrics.get("max_drawdown"), 0.0),
                    win_rate=_safe_float(metrics.get("win_rate"), 0.0),
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
