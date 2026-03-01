"""
Agent-level trade attribution and aggregate performance updates.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Dict

from loguru import logger

from data.database import AgentPerformance, Trade, get_session

_AGENT_FIELDS = {
    "TFT": "tft_score",
    "XGB": "xgb_score",
    "PPO": "ppo_score",
    "Governance": "gov_adjust",
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _compute_agent_weights(trade: Trade) -> Dict[str, float]:
    scores = {agent: _safe_float(getattr(trade, column, 0.0)) for agent, column in _AGENT_FIELDS.items()}
    total = sum(scores.values())
    if abs(total) < 1e-12:
        equal = 1.0 / len(scores)
        return {agent: equal for agent in scores}
    return {agent: score / total for agent, score in scores.items()}


def _compute_sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(max(variance, 0.0))
    if std_dev < 1e-12:
        return 0.0
    return mean_return / std_dev


def update_agent_performance(trade_or_id) -> None:
    """
    Recompute per-agent aggregate performance using all closed trades.
    Deterministic and idempotent for SQLite deployments.

    Accepts either a trade_id string or a Trade ORM object (legacy callers).
    """
    if isinstance(trade_or_id, str):
        trade_id = trade_or_id
    else:
        trade_id = getattr(trade_or_id, "trade_id", None)
    session = get_session()
    try:
        closed_trades = (
            session.query(Trade)
            .filter(Trade.status == "closed", Trade.pnl.isnot(None))
            .order_by(Trade.exit_time.asc(), Trade.id.asc())
            .all()
        )

        aggregates: Dict[str, dict] = {
            agent: {
                "total_pnl": 0.0,
                "total_trades": 0,
                "win_trades": 0,
                "returns": [],
            }
            for agent in _AGENT_FIELDS
        }

        for closed in closed_trades:
            weights = _compute_agent_weights(closed)
            pnl = _safe_float(closed.pnl)
            pnl_pct = _safe_float(closed.pnl_pct)
            for agent, weight in weights.items():
                contribution = pnl * weight
                aggregates[agent]["total_pnl"] += contribution
                aggregates[agent]["total_trades"] += 1
                if contribution > 0:
                    aggregates[agent]["win_trades"] += 1
                aggregates[agent]["returns"].append(pnl_pct * weight)

        if trade_id:
            closed_trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
            if closed_trade:
                weights = _compute_agent_weights(closed_trade)
                pnl = _safe_float(closed_trade.pnl)
                contributions = {
                    agent: pnl * weight for agent, weight in weights.items()
                }
                payload = dict(closed_trade.prediction or {})
                payload["agent_weights"] = weights
                payload["agent_contributions"] = contributions
                closed_trade.prediction = payload
                if not closed_trade.weight_snapshot_json:
                    closed_trade.weight_snapshot_json = json.dumps(weights, sort_keys=True)

        session.query(AgentPerformance).delete()
        now_iso = datetime.utcnow().isoformat()
        for agent, data in aggregates.items():
            total_trades = int(data["total_trades"])
            win_trades = int(data["win_trades"])
            total_pnl = float(data["total_pnl"])
            session.add(
                AgentPerformance(
                    agent=agent,
                    total_pnl=total_pnl,
                    total_trades=total_trades,
                    win_trades=win_trades,
                    win_rate=(win_trades / total_trades) if total_trades > 0 else 0.0,
                    sharpe=_compute_sharpe(data["returns"]),
                    avg_contribution=(total_pnl / total_trades) if total_trades > 0 else 0.0,
                    updated_at=now_iso,
                )
            )

        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error(f"Agent attribution update failed: {exc}")
    finally:
        session.close()

