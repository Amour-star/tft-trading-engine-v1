"""
Performance metrics endpoints for dashboard panels.
"""
from __future__ import annotations

import json
import numpy as np
from fastapi import APIRouter
from sqlalchemy import desc

from tft_engine.ai.metrics import max_drawdown, profit_factor, sharpe_ratio, sortino_ratio, win_rate
from tft_engine.database.connection import get_session
from tft_engine.database.models import AIMetric, ModelRegistry, Trade

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/latest")
def latest_metrics():
    session = get_session()
    try:
        latest = session.query(AIMetric).order_by(desc(AIMetric.date)).first()
        latest_xgb = (
            session.query(ModelRegistry)
            .filter(ModelRegistry.model_type == "xgboost")
            .order_by(desc(ModelRegistry.created_at))
            .first()
        )
        accuracy = 0.0
        if latest_xgb and latest_xgb.metrics_json:
            try:
                accuracy = float(json.loads(latest_xgb.metrics_json).get("accuracy", 0.0))
            except Exception:
                accuracy = 0.0

        trades = (
            session.query(Trade)
            .filter(Trade.close_timestamp.is_not(None), Trade.pnl.is_not(None))
            .order_by(Trade.close_timestamp.asc())
            .all()
        )
        pnls = [float(t.pnl or 0.0) for t in trades]
        equity = np.cumsum(pnls).tolist() if pnls else [0.0]
        pf = float(profit_factor(pnls))
        if np.isinf(pf) or np.isnan(pf):
            pf = 0.0
        return {
            "accuracy": accuracy if accuracy > 0 else float((latest.win_rate if latest else 0.0)),
            "sharpe_ratio": float(sharpe_ratio(pnls)),
            "sortino_ratio": float(sortino_ratio(pnls)),
            "max_drawdown": float(max_drawdown(equity)),
            "win_rate": float(win_rate(pnls)),
            "profit_factor": pf,
        }
    finally:
        session.close()


@router.get("/history")
def metrics_history(limit: int = 90):
    session = get_session()
    try:
        rows = session.query(AIMetric).order_by(desc(AIMetric.date)).limit(limit).all()
        return [
            {
                "date": str(r.date),
                "win_rate": float(r.win_rate),
                "avg_confidence": float(r.avg_confidence),
                "cumulative_return": float(r.cumulative_return),
                "sharpe_ratio": float(r.sharpe_ratio),
                "max_drawdown": float(r.max_drawdown),
            }
            for r in rows
        ][::-1]
    finally:
        session.close()
