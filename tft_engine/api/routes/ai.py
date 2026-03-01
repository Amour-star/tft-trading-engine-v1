"""
AI monitoring endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import desc

from tft_engine.api.runtime import get_runtime
from tft_engine.database.connection import get_session
from tft_engine.database.models import AIMetric

router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.get("/current-score")
def current_score():
    runtime = get_runtime()
    controller = runtime["ai_controller"]
    latest_pred = controller.last_prediction or {}

    session = get_session()
    try:
        latest_metric = session.query(AIMetric).order_by(desc(AIMetric.date)).first()
        win_rate = float(latest_metric.win_rate) if latest_metric else 0.0
    finally:
        session.close()

    return {
        "confidence": float(latest_pred.get("confidence", 0.0)),
        "model_version": str(latest_pred.get("model_version", "unknown")),
        "win_rate": win_rate,
    }


@router.get("/history")
def ai_history(limit: int = 90):
    session = get_session()
    try:
        rows = (
            session.query(AIMetric)
            .order_by(desc(AIMetric.date))
            .limit(limit)
            .all()
        )
        return [
            {
                "date": str(r.date),
                "win_rate": float(r.win_rate),
                "sharpe_ratio": float(r.sharpe_ratio),
                "max_drawdown": float(r.max_drawdown),
                "avg_confidence": float(r.avg_confidence),
                "cumulative_return": float(r.cumulative_return),
            }
            for r in rows
        ][::-1]
    finally:
        session.close()

