"""
FastAPI backend for the admin dashboard.
Provides REST endpoints for monitoring, control, and data access.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func, desc

from config.settings import settings
from data.database import get_session, Trade, Prediction, DailyStats, ModelMetric, LearningMetric, EngineState
from data.fetcher import KuCoinDataFetcher
from execution.paper_executor import load_paper_snapshot

app = FastAPI(title="TFT Trading Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class StatusResponse(BaseModel):
    mode: str
    engine_running: bool
    trading_enabled: bool
    paused: bool
    killed: bool
    balance: Optional[float] = None
    open_trade: Optional[Dict[str, Any]] = None
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    virtual_balance: Optional[float] = None
    paper_positions: Optional[List[Dict[str, Any]]] = None
    paper_realized_pnl: Optional[float] = None
    paper_unrealized_pnl: Optional[float] = None


class TradeResponse(BaseModel):
    trade_id: str
    pair: str
    side: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    stop_price: float
    target_price: float
    pnl: Optional[float]
    pnl_pct: Optional[float]
    r_multiple: Optional[float]
    exit_reason: Optional[str]
    confidence: Optional[float]
    status: str
    ai_reasoning: Optional[str]


class ControlRequest(BaseModel):
    action: str  # pause, resume, kill, reset_kill, force_close
    value: Optional[float] = None


class ThresholdUpdate(BaseModel):
    confidence_threshold: Optional[float] = None
    risk_per_trade: Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status", response_model=StatusResponse)
def get_status():
    session = get_session()
    try:
        mode = settings.trading.trading_mode.upper()

        # Engine running state
        running_state = session.query(EngineState).filter(EngineState.key == "running").first()
        engine_running = bool(running_state and running_state.value.get("value", False))

        paused_state = session.query(EngineState).filter(EngineState.key == "paused").first()
        paused = bool(paused_state and paused_state.value.get("value", False))

        killed_state = session.query(EngineState).filter(EngineState.key == "killed").first()
        killed = bool(killed_state and killed_state.value.get("value", False))

        # Open trade
        open_trade = session.query(Trade).filter(Trade.status == "open").first()
        open_trade_data = None
        if open_trade:
            open_trade_data = {
                "trade_id": open_trade.trade_id,
                "pair": open_trade.pair,
                "entry_price": open_trade.entry_price,
                "stop_price": open_trade.stop_price,
                "target_price": open_trade.target_price,
                "confidence": open_trade.confidence,
                "entry_time": open_trade.entry_time.isoformat(),
            }

        # Daily PnL
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        daily_result = session.query(func.sum(Trade.pnl)).filter(
            Trade.exit_time >= today_start, Trade.status == "closed"
        ).scalar()

        balance: Optional[float] = None
        virtual_balance: Optional[float] = None
        paper_positions: Optional[List[Dict[str, Any]]] = None
        paper_realized_pnl: Optional[float] = None
        paper_unrealized_pnl: Optional[float] = None

        if mode == "PAPER":
            snapshot = load_paper_snapshot()
            virtual_balance = float(snapshot.get("balance", 0.0))
            paper_realized_pnl = float(snapshot.get("realized_pnl", 0.0))
            positions = []
            unrealized = 0.0
            raw_positions = snapshot.get("positions", {})
            if raw_positions:
                fetcher = KuCoinDataFetcher()
                for symbol, p in raw_positions.items():
                    quantity = float(p.get("quantity", 0.0))
                    avg_entry = float(p.get("avg_entry_price", 0.0))
                    mark_price = avg_entry
                    position_unrealized = 0.0
                    try:
                        mark_price = float(fetcher.get_ticker(symbol)["price"])
                        position_unrealized = (mark_price - avg_entry) * quantity
                    except Exception:
                        pass

                    positions.append(
                        {
                            "symbol": symbol,
                            "quantity": quantity,
                            "avg_entry_price": avg_entry,
                            "mark_price": mark_price,
                            "unrealized_pnl": position_unrealized,
                        }
                    )
                    unrealized += position_unrealized

            paper_positions = positions
            paper_unrealized_pnl = unrealized
            balance = virtual_balance
        else:
            try:
                balance = KuCoinDataFetcher().get_balance("USDT")
            except Exception:
                balance = None

        return StatusResponse(
            mode=mode,
            engine_running=engine_running,
            trading_enabled=settings.trading.trading_enabled,
            paused=paused,
            killed=killed,
            balance=balance,
            open_trade=open_trade_data,
            daily_pnl=float(daily_result or 0),
            virtual_balance=virtual_balance,
            paper_positions=paper_positions,
            paper_realized_pnl=paper_realized_pnl,
            paper_unrealized_pnl=paper_unrealized_pnl,
        )
    finally:
        session.close()


@app.get("/api/trades", response_model=List[TradeResponse])
def get_trades(limit: int = 50, status: Optional[str] = None):
    session = get_session()
    try:
        query = session.query(Trade).order_by(desc(Trade.entry_time))
        if status:
            query = query.filter(Trade.status == status)
        trades = query.limit(limit).all()
        return [
            TradeResponse(
                trade_id=t.trade_id, pair=t.pair, side=t.side,
                entry_time=t.entry_time, exit_time=t.exit_time,
                entry_price=t.entry_price, exit_price=t.exit_price,
                stop_price=t.stop_price, target_price=t.target_price,
                pnl=t.pnl, pnl_pct=t.pnl_pct, r_multiple=t.r_multiple,
                exit_reason=t.exit_reason, confidence=t.confidence,
                status=t.status, ai_reasoning=t.ai_reasoning,
            )
            for t in trades
        ]
    finally:
        session.close()


@app.get("/api/stats")
def get_stats():
    session = get_session()
    try:
        total = session.query(Trade).filter(Trade.status == "closed").count()
        wins = session.query(Trade).filter(Trade.status == "closed", Trade.pnl > 0).count()
        total_pnl = session.query(func.sum(Trade.pnl)).filter(Trade.status == "closed").scalar() or 0
        avg_r = session.query(func.avg(Trade.r_multiple)).filter(Trade.status == "closed").scalar() or 0

        return {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": total - wins,
            "win_rate": wins / total if total > 0 else 0,
            "total_pnl": float(total_pnl),
            "avg_r_multiple": float(avg_r),
        }
    finally:
        session.close()


@app.get("/api/pnl-curve")
def get_pnl_curve(days: int = 30):
    session = get_session()
    try:
        since = datetime.utcnow() - timedelta(days=days)
        trades = (
            session.query(Trade)
            .filter(Trade.status == "closed", Trade.exit_time >= since)
            .order_by(Trade.exit_time)
            .all()
        )
        cumulative = 0.0
        curve = []
        for t in trades:
            cumulative += (t.pnl or 0)
            curve.append({
                "time": t.exit_time.isoformat(),
                "pnl": cumulative,
                "trade_id": t.trade_id,
            })
        return curve
    finally:
        session.close()


@app.get("/api/predictions")
def get_predictions(limit: int = 10):
    session = get_session()
    try:
        preds = (
            session.query(Prediction)
            .order_by(desc(Prediction.timestamp))
            .limit(limit)
            .all()
        )
        return [
            {
                "timestamp": p.timestamp.isoformat(),
                "pair": p.pair,
                "prob_up": p.prob_up,
                "prob_down": p.prob_down,
                "expected_move": p.expected_move,
                "confidence": p.confidence,
                "volatility_regime": p.volatility_regime,
                "market_regime": p.market_regime,
                "acted_on": p.acted_on,
            }
            for p in preds
        ]
    finally:
        session.close()


@app.get("/api/model-info")
def get_model_info():
    session = get_session()
    try:
        active = (
            session.query(ModelMetric)
            .filter(ModelMetric.is_active == True)
            .order_by(desc(ModelMetric.trained_at))
            .first()
        )
        if not active:
            return {"active_model": None}
        return {
            "active_model": {
                "version": active.model_version,
                "trained_at": active.trained_at.isoformat(),
                "validation_loss": active.validation_loss,
                "sharpe_ratio": active.sharpe_ratio,
                "max_drawdown": active.max_drawdown,
                "accuracy": active.accuracy,
            }
        }
    finally:
        session.close()


@app.get("/api/thresholds")
def get_thresholds():
    session = get_session()
    try:
        state = session.query(EngineState).filter(EngineState.key == "thresholds").first()
        if state:
            return state.value.get("value", {})
        return {
            "confidence_threshold": settings.trading.confidence_threshold,
            "risk_per_trade": settings.trading.risk_per_trade,
        }
    finally:
        session.close()


@app.post("/api/control")
def control_engine(req: ControlRequest):
    """Control the trading engine (pause, resume, kill, force_close)."""
    session = get_session()
    try:
        if req.action == "pause":
            state = session.query(EngineState).filter(EngineState.key == "paused").first()
            if state:
                state.value = {"value": True}
            else:
                session.add(EngineState(key="paused", value={"value": True}))
            session.commit()
            return {"status": "paused"}

        elif req.action == "resume":
            state = session.query(EngineState).filter(EngineState.key == "paused").first()
            if state:
                state.value = {"value": False}
            session.commit()
            return {"status": "resumed"}

        elif req.action == "kill":
            state = session.query(EngineState).filter(EngineState.key == "killed").first()
            if state:
                state.value = {"value": True}
            else:
                session.add(EngineState(key="killed", value={"value": True}))
            session.commit()
            return {"status": "killed"}

        elif req.action == "reset_kill":
            state = session.query(EngineState).filter(EngineState.key == "killed").first()
            if state:
                state.value = {"value": False}
            session.commit()
            return {"status": "kill_reset"}

        else:
            raise HTTPException(400, f"Unknown action: {req.action}")
    finally:
        session.close()


@app.post("/api/thresholds")
def update_thresholds(req: ThresholdUpdate):
    """Update trading thresholds."""
    session = get_session()
    try:
        state = session.query(EngineState).filter(EngineState.key == "thresholds").first()
        current = state.value.get("value", {}) if state else {}

        if req.confidence_threshold is not None:
            current["confidence_threshold"] = max(0.5, min(0.95, req.confidence_threshold))
        if req.risk_per_trade is not None:
            current["risk_per_trade"] = max(0.005, min(0.03, req.risk_per_trade))

        if state:
            state.value = {"value": current}
        else:
            session.add(EngineState(key="thresholds", value={"value": current}))
        session.commit()
        return current
    finally:
        session.close()


@app.get("/api/learning-metrics")
def get_learning_metrics(limit: int = 20):
    session = get_session()
    try:
        metrics = (
            session.query(LearningMetric)
            .order_by(desc(LearningMetric.created_at))
            .limit(limit)
            .all()
        )
        return [
            {
                "trade_id": m.trade_id,
                "forecast_accuracy": m.forecast_accuracy,
                "volatility_misread": m.volatility_misread,
                "confidence_overestimated": m.confidence_overestimated,
                "stop_too_tight": m.stop_too_tight,
                "recommended_adjustments": m.recommended_adjustments,
                "created_at": m.created_at.isoformat(),
            }
            for m in metrics
        ]
    finally:
        session.close()


def start_api():
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.dashboard.api_port)


if __name__ == "__main__":
    start_api()
