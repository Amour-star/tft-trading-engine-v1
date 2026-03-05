"""
Rolling Sharpe/Sortino calculator and metric snapshots.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from loguru import logger

from config.settings import settings
from data.database import PerformanceMetric, Trade, get_session

_ROLLING_WINDOW = 100


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_v = sum(values) / len(values)
    variance = sum((v - mean_v) ** 2 for v in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _realized_returns_from_pnl(pnls: list[float]) -> list[float]:
    returns: list[float] = []
    equity = max(_safe_float(settings.trading.paper_starting_balance, 1.0), 1e-9)
    for pnl in pnls:
        ret = pnl / equity if abs(equity) > 1e-9 else 0.0
        returns.append(_safe_float(ret, 0.0))
        equity += pnl
        if not math.isfinite(equity) or abs(equity) <= 1e-9:
            equity = max(_safe_float(settings.trading.paper_starting_balance, 1.0), 1e-9)
    return returns


def _max_drawdown_from_pnl(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    equity = settings.trading.paper_starting_balance
    peak = equity
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            drawdown = (equity - peak) / peak
        else:
            drawdown = 0.0
        if drawdown < max_dd:
            max_dd = drawdown
    return abs(max_dd)


def update_risk_metrics_snapshot(equity_override: Optional[float] = None) -> Optional[dict]:
    """
    Recompute rolling risk metrics and persist a snapshot.
    Returns snapshot payload or None when there is insufficient data.
    """
    session = get_session()
    try:
        closed_trades = (
            session.query(Trade)
            .filter(Trade.status == "closed", Trade.pnl.isnot(None))
            .order_by(Trade.exit_time.asc(), Trade.id.asc())
            .all()
        )
        if not closed_trades:
            return None

        pnls = [_safe_float(t.pnl) for t in closed_trades]
        returns = _realized_returns_from_pnl(pnls)
        rolling_returns = returns[-_ROLLING_WINDOW:]
        rolling_pnls = pnls[-_ROLLING_WINDOW:]

        mean_return = sum(rolling_returns) / len(rolling_returns)
        min_trades_for_ratios = max(2, int(getattr(settings.trading, "metrics_min_trades", 20)))
        std_return = _std(rolling_returns)
        ratios_available = len(rolling_returns) >= min_trades_for_ratios
        sharpe = (
            (mean_return / std_return)
            if ratios_available and std_return > 1e-6
            else 0.0
        )

        negative_returns = [r for r in rolling_returns if r < 0]
        std_negative = _std(negative_returns)
        sortino = (
            (mean_return / std_negative)
            if ratios_available and std_negative > 1e-6
            else 0.0
        )
        if not math.isfinite(sharpe) or abs(sharpe) > 50.0:
            sharpe = 0.0
            ratios_available = False
        if not math.isfinite(sortino) or abs(sortino) > 50.0:
            sortino = 0.0
            ratios_available = False

        wins = sum(1 for pnl in rolling_pnls if pnl > 0)
        win_rate = wins / len(rolling_pnls) if rolling_pnls else 0.0
        win_rate_display = "N/A" if len(rolling_pnls) < min_trades_for_ratios else f"{win_rate:.4f}"
        sharpe_display = "N/A" if not ratios_available else f"{sharpe:.4f}"
        sortino_display = "N/A" if not ratios_available else f"{sortino:.4f}"

        equity = (
            _safe_float(equity_override)
            if equity_override is not None
            else settings.trading.paper_starting_balance + sum(pnls)
        )
        max_drawdown = _max_drawdown_from_pnl(pnls)
        return_value = rolling_returns[-1] if rolling_returns else 0.0

        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "equity": equity,
            "return": return_value,
            "sharpe": sharpe,
            "sharpe_display": sharpe_display,
            "sortino": sortino,
            "sortino_display": sortino_display,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "win_rate_display": win_rate_display,
            "metrics_ready": bool(ratios_available),
        }
        session.add(
            PerformanceMetric(
                timestamp=snapshot["timestamp"],
                equity=snapshot["equity"],
                return_value=snapshot["return"],
                sharpe=snapshot["sharpe"],
                sortino=snapshot["sortino"],
                max_drawdown=snapshot["max_drawdown"],
                win_rate=snapshot["win_rate"],
            )
        )
        session.commit()
        return snapshot
    except Exception as exc:
        session.rollback()
        logger.error(f"Risk metric snapshot update failed: {exc}")
        return None
    finally:
        session.close()
