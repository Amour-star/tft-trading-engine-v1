"""
FastAPI backend for the admin dashboard.
Provides REST endpoints for monitoring, control, and data access.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlite3
import threading

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, inspect, text

from config.settings import XRP_ONLY_SYMBOL, settings
from data.database import (
    AgentPerformance,
    DailyStats,
    DecisionEvent,
    EngineState,
    LearningMetric,
    ModelMetric,
    PerformanceMetric,
    Prediction,
    RiskState,
    Statistics,
    StrategyParameter,
    Trade,
    get_session,
)
from data.fetcher import KuCoinDataFetcher
from engine.attribution import update_agent_performance
from engine.performance_metrics import update_risk_metrics_snapshot
from engine.strategy_evolution import StrategyEvolutionEngine
from engine.safety import SafetyManager
from execution.executor import create_executor
from execution.monitor import PositionMonitor
from execution.paper_executor import load_paper_snapshot
from models.rl_position_manager import PPOPositionManager
from models.tft_model import TFTPredictor
from risk.prop_risk_manager import PropRiskManager
from paper.reset import reset_paper_account

app = FastAPI(title="TFT Trading Engine API", version="1.0.0")
_hard_reset_lock = threading.Lock()
_hard_reset_in_progress = False

_raw_cors = [item.strip() for item in settings.dashboard.cors_origins.split(",") if item.strip()]
_cors_origins = _raw_cors or ["*"]
_allow_credentials = "*" not in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(value: Any) -> Any:
    """Recursively sanitize non-finite floats for strict JSON rendering."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


class StatusResponse(BaseModel):
    mode: str
    engine_running: bool
    trading_enabled: bool
    ai_enabled: bool
    governance_enabled: bool
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
    aggression_level: float = 1.0
    allow_shorts: bool = False
    threshold_after_scaling: Optional[float] = None
    latest_regime: Optional[Dict[str, Any]] = None


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
    ai_score: Optional[float] = None
    base_ai_score: Optional[float] = None
    tft_score: Optional[float] = None
    xgb_score: Optional[float] = None
    ppo_score: Optional[float] = None
    gov_adjust: Optional[float] = None
    final_ai_score: Optional[float] = None
    governance_code: Optional[str] = None
    status: str
    ai_reasoning: Optional[str]


class AIScoreResponse(BaseModel):
    ai_score: float
    confidence: float
    model_version: str
    win_rate: float
    timestamp: Optional[str] = None


class ControlRequest(BaseModel):
    action: str  # pause, resume, kill, reset_kill, force_close
    value: Optional[float] = None


class ThresholdUpdate(BaseModel):
    confidence_threshold: Optional[float] = None
    risk_per_trade: Optional[float] = None


class ResetPaperRequest(BaseModel):
    initial_balance: float = Field(..., gt=0.0)


def _require_admin_token(admin_token: Optional[str] = Header(None, alias="ADMIN_TOKEN")) -> str:
    expected = settings.dashboard.admin_token
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_TOKEN is not configured on the server.",
        )
    if admin_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return admin_token


def _build_monitor() -> PositionMonitor:
    fetcher = KuCoinDataFetcher()
    executor = create_executor(fetcher)
    predictor = TFTPredictor()
    rl_manager = PPOPositionManager()
    return PositionMonitor(fetcher, executor, predictor, rl_manager=rl_manager)


def _apply_post_close_updates(closed_trade_id: Optional[str], monitor: Optional[PositionMonitor] = None) -> None:
    if not closed_trade_id:
        return
    try:
        update_agent_performance(closed_trade_id)
        update_risk_metrics_snapshot()
        if monitor is not None:
            PropRiskManager(monitor.fetcher, monitor.executor).on_trade_closed_by_id(closed_trade_id)
        StrategyEvolutionEngine().evolve_if_due(open_trade_count=0)
    except Exception:
        logger.exception("Post-close maintenance hooks failed")


def _upsert_engine_state(session, key: str, payload: Dict[str, Any]) -> None:
    state = session.query(EngineState).filter(EngineState.key == key).first()
    if state:
        state.value = payload
        state.updated_at = datetime.utcnow()
    else:
        session.add(EngineState(key=key, value=payload))


def _reset_paper_sqlite(initial_balance: float) -> None:
    db_path = Path(settings.trading.paper_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                entry_fee_total REAL NOT NULL DEFAULT 0.0,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                side TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                requested_price REAL,
                fill_price REAL NOT NULL,
                fee REAL NOT NULL,
                realized_pnl REAL,
                balance_after REAL NOT NULL
            )
            """
        )
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM state")
        conn.executemany(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
            (("starting_balance", initial_balance), ("balance", initial_balance), ("realized_pnl", 0.0)),
        )
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()


@app.get("/health")
def health():
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        device = "cpu"

    session = get_session()
    try:
        running_state = session.query(EngineState).filter(EngineState.key == "running").first()
        engine_running = bool(running_state and running_state.value.get("value", False))
    finally:
        session.close()

    return {
        "status": "ok",
        "engine_running": engine_running,
        "mode": settings.trading.trading_mode.lower(),
        "device": device,
    }


@app.get("/api/status", response_model=StatusResponse)
def get_status():
    session = get_session()
    try:
        mode = settings.trading.trading_mode.upper()

        running_state = session.query(EngineState).filter(EngineState.key == "running").first()
        engine_running = bool(running_state and running_state.value.get("value", False))

        paused_state = session.query(EngineState).filter(EngineState.key == "paused").first()
        paused = bool(paused_state and paused_state.value.get("value", False))

        killed_state = session.query(EngineState).filter(EngineState.key == "killed").first()
        killed = bool(killed_state and killed_state.value.get("value", False))

        open_trade = session.query(Trade).filter(Trade.status == "open").first()
        open_trade_data = None
        if open_trade:
            pred_payload = open_trade.prediction or {}
            open_trade_data = {
                "id": open_trade.id,
                "trade_id": open_trade.trade_id,
                "pair": open_trade.pair,
                "side": open_trade.side,
                "entry_price": open_trade.entry_price,
                "stop_price": open_trade.stop_price,
                "target_price": open_trade.target_price,
                "confidence": open_trade.confidence,
                "ai_score": float(pred_payload.get("ai_score", open_trade.confidence or 0.0)),
                "ai_confidence": float(pred_payload.get("ai_confidence", open_trade.confidence or 0.0)),
                "meta_model_version": pred_payload.get("meta_model_version"),
                "tft_model_version": pred_payload.get("tft_model_version"),
                "quantity": open_trade.quantity,
                "entry_time": open_trade.entry_time.isoformat(),
                "adaptive_threshold": pred_payload.get("adaptive_threshold"),
                "regime": pred_payload.get("regime"),
            }

        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        daily_result = session.query(func.sum(Trade.pnl)).filter(
            Trade.exit_time >= today_start,
            Trade.status == "closed",
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
                            "side": "LONG" if quantity > 0 else "SHORT",
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

        latest_decision = (
            session.query(DecisionEvent)
            .order_by(desc(DecisionEvent.timestamp), desc(DecisionEvent.id))
            .first()
        )
        threshold_after_scaling = (
            float(latest_decision.adaptive_threshold)
            if latest_decision and latest_decision.adaptive_threshold is not None
            else None
        )
        latest_regime = None
        if latest_decision:
            latest_regime = {
                "trend": latest_decision.regime,
                "volatility": latest_decision.volatility_regime,
            }

        return StatusResponse(
            mode=mode,
            engine_running=engine_running,
            trading_enabled=settings.trading.trading_enabled,
            ai_enabled=settings.runtime.ai_enabled,
            governance_enabled=settings.governance.llm_enabled,
            paused=paused,
            killed=killed,
            balance=balance,
            open_trade=open_trade_data,
            daily_pnl=float(daily_result or 0.0),
            virtual_balance=virtual_balance,
            paper_positions=paper_positions,
            paper_realized_pnl=paper_realized_pnl,
            paper_unrealized_pnl=paper_unrealized_pnl,
            aggression_level=float(settings.trading.aggression_level),
            allow_shorts=bool(settings.trading.allow_shorts),
            threshold_after_scaling=threshold_after_scaling,
            latest_regime=latest_regime,
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
                trade_id=t.trade_id,
                pair=t.pair,
                side=t.side,
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                stop_price=t.stop_price,
                target_price=t.target_price,
                pnl=t.pnl,
                pnl_pct=t.pnl_pct,
                r_multiple=t.r_multiple,
                exit_reason=t.exit_reason,
                confidence=t.confidence,
                ai_score=t.ai_score,
                base_ai_score=t.base_ai_score,
                tft_score=t.tft_score,
                xgb_score=t.xgb_score,
                ppo_score=t.ppo_score,
                gov_adjust=t.gov_adjust,
                final_ai_score=t.final_ai_score,
                governance_code=t.governance_code,
                status=t.status,
                ai_reasoning=t.ai_reasoning,
            )
            for t in trades
        ]
    finally:
        session.close()


@app.post("/api/trades/{trade_id}/force_close")
def force_close_trade(trade_id: str):
    monitor = _build_monitor()
    closed_id = monitor.force_close_trade(trade_id)
    if not closed_id:
        raise HTTPException(status_code=404, detail=f"Open trade not found: {trade_id}")
    _apply_post_close_updates(closed_id, monitor=monitor)
    return {"status": "force_closed", "trade_id": closed_id}


@app.post("/reconcile")
def enqueue_reconciliation(symbol: str = XRP_ONLY_SYMBOL):
    target_symbol = str(symbol or XRP_ONLY_SYMBOL).strip() or XRP_ONLY_SYMBOL
    fetcher = KuCoinDataFetcher()
    executor = create_executor(fetcher)
    queued = executor.schedule_reconciliation([target_symbol], source="api")
    event_id = str(queued[0].id) if queued else None
    return {
        "status": "queued",
        "symbol": target_symbol,
        "event_id": event_id,
    }


@app.get("/api/stats")
def get_stats():
    session = get_session()
    try:
        total = session.query(Trade).filter(Trade.status == "closed").count()
        wins = session.query(Trade).filter(Trade.status == "closed", Trade.pnl > 0).count()
        long_trades = session.query(Trade).filter(Trade.status == "closed", Trade.side == "BUY").count()
        short_trades = session.query(Trade).filter(Trade.status == "closed", Trade.side == "SELL").count()
        total_pnl = session.query(func.sum(Trade.pnl)).filter(Trade.status == "closed").scalar() or 0
        avg_r = session.query(func.avg(Trade.r_multiple)).filter(Trade.status == "closed").scalar() or 0

        return {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": total - wins,
            "long_trades": long_trades,
            "short_trades": short_trades,
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
            curve.append(
                {
                    "time": t.exit_time.isoformat(),
                    "pnl": cumulative,
                    "trade_id": t.trade_id,
                }
            )
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


@app.get("/api/ai/current-score", response_model=AIScoreResponse)
def get_ai_current_score():
    session = get_session()
    try:
        latest_pred = (
            session.query(Prediction)
            .order_by(desc(Prediction.timestamp))
            .first()
        )

        total = session.query(Trade).filter(Trade.status == "closed").count()
        wins = session.query(Trade).filter(Trade.status == "closed", Trade.pnl > 0).count()
        win_rate = wins / total if total > 0 else 0.0

        if latest_pred:
            return AIScoreResponse(
                ai_score=float(latest_pred.confidence or 0.0),
                confidence=float(latest_pred.confidence or 0.0),
                model_version=str(latest_pred.model_version or "unknown"),
                win_rate=float(win_rate),
                timestamp=latest_pred.timestamp.isoformat(),
            )

        return AIScoreResponse(
            ai_score=0.0,
            confidence=0.0,
            model_version="unknown",
            win_rate=float(win_rate),
            timestamp=None,
        )
    finally:
        session.close()


@app.get("/api/ai/history")
def get_ai_history(limit: int = 180):
    session = get_session()
    try:
        rows = (
            session.query(Prediction)
            .order_by(desc(Prediction.timestamp))
            .limit(limit)
            .all()
        )
        return [
            {
                "timestamp": p.timestamp.isoformat(),
                "pair": p.pair,
                "ai_score": float(p.confidence or 0.0),
                "confidence": float(p.confidence or 0.0),
                "prob_up": float(p.prob_up or 0.0),
                "prob_down": float(p.prob_down or 0.0),
                "expected_move": float(p.expected_move or 0.0),
                "model_version": p.model_version or "unknown",
            }
            for p in reversed(rows)
        ]
    finally:
        session.close()


@app.get("/api/ai/analytics")
def get_ai_analytics(days: int = 7):
    lookback_days = max(1, min(int(days), 365))
    since = datetime.utcnow() - timedelta(days=lookback_days)
    session = get_session()
    try:
        trades = (
            session.query(Trade)
            .filter(Trade.status == "closed", Trade.exit_time >= since)
            .order_by(Trade.exit_time.asc())
            .all()
        )

        timeline: list[dict[str, Any]] = []
        confidence_values: list[float] = []
        cumulative = 0.0
        peak = 0.0

        for trade in trades:
            pred_payload = trade.prediction or {}
            base_score = float(
                trade.base_ai_score
                if trade.base_ai_score is not None
                else pred_payload.get("base_ai_score", trade.confidence or 0.0)
            )
            final_score = float(
                trade.ai_score
                if trade.ai_score is not None
                else pred_payload.get("ai_score", trade.confidence or 0.0)
            )
            confidence = float(trade.confidence or final_score)
            pnl = float(trade.pnl or 0.0)
            cumulative += pnl
            peak = max(peak, cumulative)
            drawdown = cumulative - peak
            confidence_values.append(confidence)

            timeline.append(
                {
                    "timestamp": (trade.exit_time or trade.entry_time).isoformat(),
                    "pair": trade.pair,
                    "before_score": base_score,
                    "final_score": final_score,
                    "confidence": confidence,
                    "pnl": pnl,
                    "equity_curve": cumulative,
                    "drawdown": drawdown,
                    "governance_code": trade.governance_code
                    or str(pred_payload.get("governance_code", "")),
                }
            )

        avg_before = sum(row["before_score"] for row in timeline) / len(timeline) if timeline else 0.0
        avg_final = sum(row["final_score"] for row in timeline) / len(timeline) if timeline else 0.0
        avg_delta = avg_final - avg_before

        return {
            "days": lookback_days,
            "points": len(timeline),
            "timeline": timeline,
            "confidence_values": confidence_values,
            "impact": {
                "avg_before": avg_before,
                "avg_final": avg_final,
                "avg_delta": avg_delta,
            },
        }
    finally:
        session.close()


@app.get("/api/agent-attribution")
def get_agent_attribution():
    session = get_session()
    try:
        rows = session.query(AgentPerformance).order_by(AgentPerformance.agent.asc()).all()
        return [
            {
                "agent": r.agent,
                "total_pnl": float(r.total_pnl or 0.0),
                "total_trades": int(r.total_trades or 0),
                "win_trades": int(r.win_trades or 0),
                "win_rate": float(r.win_rate or 0.0),
                "sharpe": float(r.sharpe or 0.0),
                "avg_contribution": float(r.avg_contribution or 0.0),
                "updated_at": r.updated_at,
            }
            for r in rows
        ]
    finally:
        session.close()


@app.get("/api/risk-metrics")
def get_risk_metrics(limit: int = 200):
    session = get_session()
    try:
        rows = (
            session.query(PerformanceMetric)
            .order_by(desc(PerformanceMetric.id))
            .limit(max(1, min(int(limit), 2000)))
            .all()
        )
        return [
            {
                "timestamp": r.timestamp,
                "equity": float(r.equity or 0.0),
                "return": float(r.return_value or 0.0),
                "sharpe": float(r.sharpe or 0.0),
                "sortino": float(r.sortino or 0.0),
                "max_drawdown": float(r.max_drawdown or 0.0),
                "win_rate": float(r.win_rate or 0.0),
            }
            for r in reversed(rows)
        ]
    finally:
        session.close()


@app.get("/api/strategy-evolution")
def get_strategy_evolution(limit: int = 180):
    session = get_session()
    try:
        rows = (
            session.query(StrategyParameter)
            .order_by(desc(StrategyParameter.id))
            .limit(max(1, min(int(limit), 2000)))
            .all()
        )
        return [
            {
                "timestamp": r.timestamp,
                "tft_weight": float(r.tft_weight or 0.0),
                "xgb_weight": float(r.xgb_weight or 0.0),
                "ppo_weight": float(r.ppo_weight or 0.0),
                "confidence_threshold": float(r.confidence_threshold or 0.0),
                "risk_per_trade": float(r.risk_per_trade or 0.0),
            }
            for r in reversed(rows)
        ]
    finally:
        session.close()


@app.get("/api/risk-state")
def get_risk_state(limit: int = 30):
    session = get_session()
    try:
        rows = (
            session.query(RiskState)
            .order_by(desc(RiskState.date))
            .limit(max(1, min(int(limit), 365)))
            .all()
        )
        return [
            {
                "date": r.date,
                "trading_enabled": bool(r.trading_enabled),
                "daily_loss": float(r.daily_loss or 0.0),
                "max_equity": float(r.max_equity or 0.0),
                "consecutive_losses": int(r.consecutive_losses or 0),
            }
            for r in rows
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
        defaults = {
            "confidence_threshold": settings.trading.confidence_threshold,
            "min_confidence": settings.trading.confidence_threshold,
            "aggression_level": settings.trading.aggression_level,
            "allow_shorts": settings.trading.allow_shorts,
            "max_open_trades": settings.trading.max_open_trades,
            "spread_max_pct": settings.trading.max_spread_pct,
            "risk_per_trade": settings.trading.risk_per_trade,
        }
        state = session.query(EngineState).filter(EngineState.key == "thresholds").first()
        if state:
            payload = state.value.get("value", {})
            if isinstance(payload, dict):
                return {**defaults, **payload}
        return defaults
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

        if req.action == "resume":
            state = session.query(EngineState).filter(EngineState.key == "paused").first()
            if state:
                state.value = {"value": False}
            session.commit()
            return {"status": "resumed"}

        if req.action == "kill":
            state = session.query(EngineState).filter(EngineState.key == "killed").first()
            if state:
                state.value = {"value": True}
            else:
                session.add(EngineState(key="killed", value={"value": True}))
            session.commit()
            return {"status": "killed"}

        if req.action == "reset_kill":
            state = session.query(EngineState).filter(EngineState.key == "killed").first()
            if state:
                state.value = {"value": False}
            session.commit()
            return {"status": "kill_reset"}

        if req.action == "reset_safe_mode":
            safe = session.query(EngineState).filter(EngineState.key == "safe_mode").first()
            if safe:
                safe.value = {"value": False}
            paused = session.query(EngineState).filter(EngineState.key == "paused").first()
            if paused:
                paused.value = {"value": False}
            session.commit()
            return {"status": "safe_mode_reset"}

        if req.action == "force_close":
            monitor = _build_monitor()
            monitor.force_close_all()
            latest = (
                session.query(Trade)
                .filter(Trade.status == "closed", Trade.pnl.isnot(None))
                .order_by(desc(Trade.exit_time), desc(Trade.id))
                .first()
            )
            _apply_post_close_updates(latest.trade_id if latest else None, monitor=monitor)
            return {"status": "force_closed"}

        raise HTTPException(400, f"Unknown action: {req.action}")
    finally:
        session.close()


@app.post("/admin/reset-paper")
def reset_paper_account_endpoint(
    req: ResetPaperRequest,
    _admin_token: str = Depends(_require_admin_token),
):
    """Reset the paper account via authenticated API calls."""
    try:
        summary = reset_paper_account(initial_balance=req.initial_balance, confirm=True)
    except Exception as exc:
        logger.exception("Paper reset via API failed: {}", exc)
        raise HTTPException(status_code=500, detail="Paper reset failed")

    return {
        "ok": True,
        "initial_balance": summary.initial_balance,
        "deleted": summary.deleted,
        "updated": summary.updated,
        "sqlite": summary.sqlite_deleted,
    }


@app.post("/api/admin/hard-reset")
def hard_reset_endpoint(
    background_tasks: BackgroundTasks,
    _admin_token: str = Depends(_require_admin_token),
):
    global _hard_reset_in_progress
    with _hard_reset_lock:
        if _hard_reset_in_progress:
            raise HTTPException(status_code=409, detail="Hard reset already in progress")
        _hard_reset_in_progress = True

    initial_balance = float(settings.trading.paper_initial_balance)

    def _run_hard_reset() -> None:
        global _hard_reset_in_progress
        safety = SafetyManager()
        try:
            safety.pause_trading()
            _reset_paper_sqlite(initial_balance)

            session = get_session()
            deleted_counts: Dict[str, int] = {}
            try:
                with session.begin():
                    inspector = inspect(session.bind)
                    targets = [
                        "trades",
                        "positions",
                        "predictions",
                        "pnl_history",
                        "equity_curve",
                        "agent_logs",
                    ]
                    for table in targets:
                        if inspector.has_table(table):
                            result = session.execute(text(f"DELETE FROM {table}"))
                            deleted_counts[table] = int(result.rowcount or 0)
                        else:
                            deleted_counts[table] = 0

                    stats_record = session.query(Statistics).first()
                    if not stats_record:
                        stats_record = Statistics()
                        session.add(stats_record)
                    stats_record.total_trades = 0
                    stats_record.wins = 0
                    stats_record.losses = 0
                    stats_record.win_rate = 0.0
                    stats_record.avg_r = 0.0
                    stats_record.total_pnl = 0.0
                    stats_record.unrealized_pnl = 0.0

                    stamp = datetime.utcnow().isoformat()
                    _upsert_engine_state(session, "open_positions", {"value": []})
                    _upsert_engine_state(session, "trade_history", {"value": []})
                    _upsert_engine_state(
                        session,
                        "runtime_stats",
                        {
                            "value": {
                                "total_trades": 0,
                                "wins": 0,
                                "losses": 0,
                                "win_rate": 0.0,
                                "avg_r": 0.0,
                                "total_pnl": 0.0,
                                "unrealized_pnl": 0.0,
                            }
                        },
                    )
                    _upsert_engine_state(
                        session,
                        "paper_reset_pending",
                        {"value": True, "initial_balance": initial_balance, "timestamp": stamp},
                    )
                    _upsert_engine_state(session, "hard_reset_pending", {"value": True, "timestamp": stamp})
                    _upsert_engine_state(session, "safe_mode", {"value": False})
                    _upsert_engine_state(session, "killed", {"value": False})
                    _upsert_engine_state(session, "paused", {"value": False})

                logger.bind(event="HARD_RESET_COMPLETED", deleted=deleted_counts).info("HARD_RESET_COMPLETED")
            finally:
                session.close()
        except Exception:
            logger.exception("Hard reset background task failed")
        finally:
            try:
                safety.resume_trading()
            except Exception:
                logger.exception("Failed to resume trading after hard reset")
            with _hard_reset_lock:
                _hard_reset_in_progress = False

    background_tasks.add_task(_run_hard_reset)
    return {"status": "reset started", "initial_balance": initial_balance}


@app.post("/api/thresholds")
def update_thresholds(req: ThresholdUpdate):
    """Update trading thresholds."""
    session = get_session()
    try:
        state = session.query(EngineState).filter(EngineState.key == "thresholds").first()
        current = state.value.get("value", {}) if state else {}

        if req.confidence_threshold is not None:
            current["confidence_threshold"] = max(0.40, min(0.95, req.confidence_threshold))
            current["min_confidence"] = current["confidence_threshold"]
        if req.risk_per_trade is not None:
            current["risk_per_trade"] = max(0.002, min(0.05, req.risk_per_trade))

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


@app.get("/api/decision-events")
def get_decision_events(limit: int = 50):
    """Return recent decision cycle events (including NO_TRADE cycles)."""
    session = get_session()
    try:
        rows = (
            session.query(DecisionEvent)
            .order_by(desc(DecisionEvent.timestamp))
            .limit(max(1, min(int(limit), 500)))
            .all()
        )
        return [
            {
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "mode": r.mode,
                "status": r.status,
                "reason": r.reason,
                "candidates_evaluated": r.candidates_evaluated,
                "candidates_valid": r.candidates_valid,
                "best_pair": r.best_pair,
                "best_score": _json_safe(r.best_score),
                "best_ai_score": _json_safe(r.best_ai_score),
                "best_confidence": _json_safe(r.best_confidence),
                "best_prob_up": _json_safe(r.best_prob_up),
                "best_prob_down": _json_safe(r.best_prob_down),
                "regime": r.regime,
                "volatility_regime": r.volatility_regime,
                "adaptive_threshold": _json_safe(r.adaptive_threshold),
                "top_candidates": _json_safe(r.top_candidates_json),
            }
            for r in rows
        ]
    finally:
        session.close()


def start_api():
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run(app, host=settings.dashboard.api_host, port=settings.dashboard.api_port)


if __name__ == "__main__":
    start_api()
