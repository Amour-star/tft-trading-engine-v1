"""
FastAPI backend for the admin dashboard.
Provides REST endpoints for monitoring, control, and data access.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlite3
import threading

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, inspect, text

from config.settings import ACTIVE_SYMBOL, settings
from data.database import (
    AgentPerformance,
    DailyStats,
    DecisionEvent,
    EquityHistory,
    EngineState,
    LearningMetric,
    MetricSnapshot,
    ModelMetric,
    Position,
    PerformanceMetric,
    Prediction,
    RiskState,
    SignalRecord,
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
from risk.safety_layer import request_kill_switch_rebaseline
from risk.prop_risk_manager import PropRiskManager, request_prop_risk_rebaseline
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
        return value if math.isfinite(value) else 0.0
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return numeric


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _is_symbol_scoped_runtime() -> bool:
    return not _quant_engine_enabled()


def _normalized_symbol(symbol: Optional[str]) -> str:
    return str(symbol or ACTIVE_SYMBOL).strip().upper()


def _apply_trade_symbol_scope(query, symbol: Optional[str] = None):
    if symbol:
        return query.filter(Trade.pair == _normalized_symbol(symbol))
    if _is_symbol_scoped_runtime():
        return query.filter(Trade.pair == ACTIVE_SYMBOL)
    return query


def _load_engine_state_value(session, key: str, default: Any = None) -> Any:
    row = session.query(EngineState).filter(EngineState.key == key).first()
    if not row or not isinstance(row.value, dict):
        return default
    return row.value.get("value", default)


class StatusResponse(BaseModel):
    mode: str
    engine_running: bool
    trading_enabled: bool
    accept_new_trades: bool
    ai_enabled: bool
    governance_enabled: bool
    paused: bool
    killed: bool
    market_data_source: str = "public_ticker"
    ticker_source: str = "unknown"
    orderbook_source: str = "unknown"
    synthetic_active: bool = False
    market_data_warning: Optional[str] = None
    credentials_present: bool = False
    auth_required: bool = False
    auth_valid: Optional[bool] = None
    auth_error: str = ""
    balance: float = 0.0
    open_trade: Optional[Dict[str, Any]] = None
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    virtual_balance: float = 0.0
    paper_positions: List[Dict[str, Any]] = Field(default_factory=list)
    paper_realized_pnl: float = 0.0
    paper_unrealized_pnl: float = 0.0
    paper_position_value: float = 0.0
    paper_equity: float = 0.0
    aggression_level: float = 1.0
    allow_shorts: bool = False
    threshold_after_scaling: float = 0.0
    latest_regime: Optional[Dict[str, Any]] = None
    universe: List[str] = Field(default_factory=list)


class TradeResponse(BaseModel):
    trade_id: str
    pair: str
    side: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: float = 0.0
    stop_price: float
    target_price: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    exit_reason: Optional[str]
    confidence: float = 0.0
    ai_score: float = 0.0
    base_ai_score: float = 0.0
    tft_score: float = 0.0
    xgb_score: float = 0.0
    ppo_score: float = 0.0
    gov_adjust: float = 0.0
    final_ai_score: float = 0.0
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


def _quant_engine_enabled() -> bool:
    return os.getenv("QUANT_ENGINE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


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
        quant_enabled = _quant_engine_enabled()

        running_state = session.query(EngineState).filter(EngineState.key == "running").first()
        engine_running = bool(running_state and running_state.value.get("value", False))

        paused_state = session.query(EngineState).filter(EngineState.key == "paused").first()
        paused = bool(paused_state and paused_state.value.get("value", False))

        killed_state = session.query(EngineState).filter(EngineState.key == "killed").first()
        killed = bool(killed_state and killed_state.value.get("value", False))
        accept_new_trades = bool(_load_engine_state_value(session, "accept_new_trades", True))
        market_data_state = _load_engine_state_value(session, "market_data_status", {}) or {}
        if not isinstance(market_data_state, dict):
            market_data_state = {}
        market_data_source = str(market_data_state.get("market_data_source", "public_ticker"))
        ticker_source = str(market_data_state.get("ticker_source", "unknown"))
        orderbook_source = str(market_data_state.get("orderbook_source", "unknown"))
        synthetic_active = bool(market_data_state.get("synthetic_active", False))
        market_data_warning = ""
        if synthetic_active:
            market_data_warning = (
                "Synthetic market data is active. Trading values may be unrealistic until real feeds recover."
            )

        open_trade_query = session.query(Trade).filter(Trade.status == "open")
        open_trade_query = _apply_trade_symbol_scope(open_trade_query)
        open_trade = open_trade_query.order_by(desc(Trade.entry_time), desc(Trade.id)).first()
        open_trade_data = None

        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        daily_query = session.query(func.sum(Trade.pnl)).filter(
            Trade.exit_time >= today_start,
            Trade.status == "closed",
        )
        daily_query = _apply_trade_symbol_scope(daily_query)
        daily_result = daily_query.scalar()

        balance = 0.0
        virtual_balance = 0.0
        paper_positions: List[Dict[str, Any]] = []
        paper_realized_pnl = 0.0
        paper_unrealized_pnl = 0.0
        paper_position_value = 0.0
        paper_starting_balance = _safe_float(settings.trading.paper_starting_balance, 0.0)
        paper_equity = 0.0
        universe: List[str] = []

        if mode == "PAPER" and quant_enabled:
            fetcher = KuCoinDataFetcher()
            quant_wallet = session.query(EngineState).filter(EngineState.key == "quant_wallet").first()
            quant_runtime = session.query(EngineState).filter(EngineState.key == "quant_runtime").first()
            wallet_payload = quant_wallet.value if quant_wallet and isinstance(quant_wallet.value, dict) else {}
            runtime_payload = quant_runtime.value if quant_runtime and isinstance(quant_runtime.value, dict) else {}
            risk_payload = runtime_payload.get("risk", {}) if isinstance(runtime_payload, dict) else {}
            virtual_balance = _safe_float(wallet_payload.get("cash_balance"), paper_starting_balance)
            paper_realized_pnl = _safe_float(wallet_payload.get("realized_pnl"), 0.0)
            paper_unrealized_pnl = _safe_float(runtime_payload.get("unrealized_pnl"), 0.0)
            paper_equity = _safe_float(runtime_payload.get("equity"), virtual_balance + paper_unrealized_pnl)
            balance = virtual_balance

            positions_rows = session.query(Position).all()
            for row in positions_rows:
                quantity = _safe_float(row.quantity, 0.0)
                avg_entry = _safe_float(row.avg_entry_price, 0.0)
                mark_price = avg_entry
                try:
                    mark_price = _safe_float(fetcher.get_ticker(row.symbol).get("price"), avg_entry)
                except Exception:
                    pass
                unrealized = (mark_price - avg_entry) * quantity
                paper_positions.append(
                    {
                        "symbol": row.symbol,
                        "quantity": quantity,
                        "side": "LONG" if quantity >= 0 else "SHORT",
                        "avg_entry_price": avg_entry,
                        "mark_price": mark_price,
                        "unrealized_pnl": _safe_float(unrealized, 0.0),
                    }
                )
                paper_position_value += abs(mark_price * quantity)
                universe.append(str(row.symbol))
            if not universe:
                universe = sorted(
                    {
                        str(item[0])
                        for item in session.query(SignalRecord.pair).distinct().limit(32).all()
                        if item and item[0]
                    }
                )
            if not universe:
                env_uni = os.getenv("UNIVERSE", "")
                universe = [token.strip().upper() for token in env_uni.split(",") if token.strip()]
            universe = sorted(set(universe))
        elif mode == "PAPER":
            snapshot = load_paper_snapshot()
            paper_starting_balance = _safe_float(snapshot.get("starting_balance"), paper_starting_balance)
            virtual_balance = _safe_float(snapshot.get("balance"), 0.0)
            paper_realized_pnl = _safe_float(snapshot.get("realized_pnl"), 0.0)
            positions = []
            unrealized = 0.0
            raw_positions = snapshot.get("positions", {}) or {}
            if raw_positions:
                fetcher = KuCoinDataFetcher()
                for symbol, p in raw_positions.items():
                    quantity = _safe_float(p.get("quantity"), 0.0)
                    avg_entry = _safe_float(p.get("avg_entry_price"), 0.0)
                    mark_price = avg_entry
                    position_unrealized = 0.0
                    try:
                        mark_price = _safe_float(fetcher.get_ticker(symbol).get("price"), avg_entry)
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
                    paper_position_value += mark_price * quantity

            paper_positions = positions
            paper_unrealized_pnl = _safe_float(unrealized, 0.0)
            paper_equity = _safe_float(paper_starting_balance + paper_realized_pnl + paper_unrealized_pnl, 0.0)
            balance = virtual_balance
            universe = sorted({str(pos.get("symbol")) for pos in positions if pos.get("symbol")})
        else:
            try:
                balance = _safe_float(KuCoinDataFetcher().get_balance("USDT"), 0.0)
            except Exception:
                balance = 0.0

        if quant_enabled:
            latest_signal = (
                session.query(SignalRecord)
                .order_by(desc(SignalRecord.timestamp), desc(SignalRecord.id))
                .first()
            )
            threshold_after_scaling = _safe_float(
                ((latest_signal.payload or {}).get("threshold") if latest_signal else None),
                _safe_float(settings.trading.confidence_threshold, 0.0),
            )
            latest_regime = None
            if latest_signal:
                latest_regime = {"trend": latest_signal.regime, "volatility": latest_signal.regime}
        else:
            latest_decision = (
                session.query(DecisionEvent)
                .order_by(desc(DecisionEvent.timestamp), desc(DecisionEvent.id))
                .first()
            )
            threshold_after_scaling = _safe_float(
                latest_decision.adaptive_threshold if latest_decision else None,
                0.0,
            )
            latest_regime = None
            if latest_decision:
                latest_regime = {
                    "trend": latest_decision.regime,
                    "volatility": latest_decision.volatility_regime,
                }

        if open_trade:
            pred_payload = open_trade.prediction or {}
            confidence_fallback = _safe_float(open_trade.confidence, 0.0)
            open_trade_data = {
                "id": open_trade.id,
                "trade_id": open_trade.trade_id,
                "pair": open_trade.pair,
                "side": open_trade.side,
                "entry_price": _safe_float(open_trade.entry_price, 0.0),
                "stop_price": _safe_float(open_trade.stop_price, 0.0),
                "target_price": _safe_float(open_trade.target_price, 0.0),
                "confidence": confidence_fallback,
                "ai_score": _safe_float(pred_payload.get("ai_score"), confidence_fallback),
                "ai_confidence": _safe_float(pred_payload.get("ai_confidence"), confidence_fallback),
                "meta_model_version": pred_payload.get("meta_model_version"),
                "tft_model_version": pred_payload.get("tft_model_version"),
                "quantity": _safe_float(open_trade.quantity, 0.0),
                "entry_time": (open_trade.entry_time or datetime.utcnow()).isoformat(),
                "adaptive_threshold": _safe_float(pred_payload.get("adaptive_threshold"), 0.0),
                "regime": pred_payload.get("regime"),
            }

        # In PAPER mode, the paper wallet snapshot is the source of truth for open positions.
        if mode == "PAPER":
            if not paper_positions:
                open_trade_data = None
            else:
                primary_position = paper_positions[0]
                primary_symbol = str(primary_position.get("symbol") or ACTIVE_SYMBOL)
                if not open_trade_data or str(open_trade_data.get("pair")) != primary_symbol:
                    open_trade_data = {
                        "id": None,
                        "trade_id": None,
                        "pair": primary_symbol,
                        "side": (
                            "SELL"
                            if str(primary_position.get("side") or "").upper() == "SHORT"
                            else "BUY"
                        ),
                        "entry_price": _safe_float(primary_position.get("avg_entry_price"), 0.0),
                        "stop_price": 0.0,
                        "target_price": 0.0,
                        "confidence": 0.0,
                        "ai_score": 0.0,
                        "ai_confidence": 0.0,
                        "meta_model_version": None,
                        "tft_model_version": None,
                        "quantity": _safe_float(primary_position.get("quantity"), 0.0),
                        "entry_time": datetime.utcnow().isoformat(),
                        "adaptive_threshold": 0.0,
                        "regime": None,
                    }

        return StatusResponse(
            mode=mode,
            engine_running=engine_running,
            trading_enabled=bool(settings.trading.trading_enabled and accept_new_trades),
            accept_new_trades=accept_new_trades,
            ai_enabled=settings.runtime.ai_enabled,
            governance_enabled=settings.governance.llm_enabled,
            paused=paused,
            killed=killed,
            market_data_source=market_data_source,
            ticker_source=ticker_source,
            orderbook_source=orderbook_source,
            synthetic_active=synthetic_active,
            market_data_warning=market_data_warning or None,
            credentials_present=bool(market_data_state.get("credentials_present", False)),
            auth_required=bool(market_data_state.get("auth_required", False)),
            auth_valid=market_data_state.get("auth_valid"),
            auth_error=str(market_data_state.get("auth_error", "")),
            balance=balance,
            open_trade=open_trade_data,
            daily_pnl=_safe_float(daily_result, 0.0),
            virtual_balance=virtual_balance,
            paper_positions=paper_positions,
            paper_realized_pnl=paper_realized_pnl,
            paper_unrealized_pnl=paper_unrealized_pnl,
            paper_position_value=_safe_float(paper_position_value, 0.0),
            paper_equity=_safe_float(paper_equity, 0.0),
            aggression_level=_safe_float(settings.trading.aggression_level, 1.0),
            allow_shorts=bool(settings.trading.allow_shorts),
            threshold_after_scaling=threshold_after_scaling,
            latest_regime=latest_regime,
            universe=universe,
        )
    finally:
        session.close()


@app.get("/api/trades", response_model=List[TradeResponse])
def get_trades(limit: int = 50, status: Optional[str] = None, symbol: Optional[str] = None):
    session = get_session()
    try:
        query = session.query(Trade).order_by(desc(Trade.entry_time))
        query = _apply_trade_symbol_scope(query, symbol=symbol)
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
                entry_price=_safe_float(t.entry_price, 0.0),
                exit_price=_safe_float(t.exit_price, 0.0),
                stop_price=_safe_float(t.stop_price, 0.0),
                target_price=_safe_float(t.target_price, 0.0),
                pnl=_safe_float(t.pnl, 0.0),
                pnl_pct=_safe_float(t.pnl_pct, 0.0),
                r_multiple=_safe_float(t.r_multiple, 0.0),
                exit_reason=t.exit_reason,
                confidence=_safe_float(t.confidence, 0.0),
                ai_score=_safe_float(t.ai_score, 0.0),
                base_ai_score=_safe_float(t.base_ai_score, 0.0),
                tft_score=_safe_float(t.tft_score, 0.0),
                xgb_score=_safe_float(t.xgb_score, 0.0),
                ppo_score=_safe_float(t.ppo_score, 0.0),
                gov_adjust=_safe_float(t.gov_adjust, 0.0),
                final_ai_score=_safe_float(t.final_ai_score, 0.0),
                governance_code=t.governance_code,
                status=t.status or "open",
                ai_reasoning=t.ai_reasoning,
            )
            for t in trades
        ]
    finally:
        session.close()


@app.get("/api/positions")
def get_positions(symbol: Optional[str] = None):
    mode = settings.trading.trading_mode.upper()
    quant_enabled = _quant_engine_enabled()
    target_symbol = _normalized_symbol(symbol) if symbol else None
    session = get_session()
    try:
        rows: List[Dict[str, Any]] = []
        if mode == "PAPER" and quant_enabled:
            fetcher = KuCoinDataFetcher()
            for row in session.query(Position).all():
                quantity = _safe_float(row.quantity, 0.0)
                avg_entry_price = _safe_float(row.avg_entry_price, 0.0)
                mark_price = avg_entry_price
                try:
                    mark_price = _safe_float(fetcher.get_ticker(row.symbol).get("price"), avg_entry_price)
                except Exception:
                    pass
                unrealized = (mark_price - avg_entry_price) * quantity
                rows.append(
                    {
                        "symbol": row.symbol,
                        "side": "LONG" if quantity >= 0 else "SHORT",
                        "quantity": quantity,
                        "entry_price": avg_entry_price,
                        "avg_entry_price": avg_entry_price,
                        "mark_price": mark_price,
                        "notional": abs(quantity * mark_price),
                        "unrealized_pnl": _safe_float(unrealized, 0.0),
                    }
                )
            if target_symbol:
                return [row for row in rows if str(row.get("symbol", "")).upper() == target_symbol]
            return rows

        if mode == "PAPER":
            snapshot = load_paper_snapshot()
            fetcher = KuCoinDataFetcher()
            for symbol, payload in (snapshot.get("positions", {}) or {}).items():
                quantity = _safe_float(payload.get("quantity"), 0.0)
                avg_entry_price = _safe_float(payload.get("avg_entry_price"), 0.0)
                mark_price = avg_entry_price
                try:
                    mark_price = _safe_float(fetcher.get_ticker(symbol).get("price"), avg_entry_price)
                except Exception:
                    pass
                unrealized = (mark_price - avg_entry_price) * quantity
                rows.append(
                    {
                        "symbol": symbol,
                        "side": "LONG" if quantity >= 0 else "SHORT",
                        "quantity": quantity,
                        "entry_price": avg_entry_price,
                        "avg_entry_price": avg_entry_price,
                        "mark_price": mark_price,
                        "notional": abs(quantity * mark_price),
                        "unrealized_pnl": _safe_float(unrealized, 0.0),
                    }
                )
            if target_symbol:
                return [row for row in rows if str(row.get("symbol", "")).upper() == target_symbol]
            return rows

        fetcher = KuCoinDataFetcher()
        open_trades_query = session.query(Trade).filter(Trade.status == "open")
        open_trades_query = _apply_trade_symbol_scope(open_trades_query, symbol=symbol)
        open_trades = open_trades_query.all()
        for trade in open_trades:
            side = str(trade.side or "BUY").upper()
            qty = _safe_float(trade.quantity, 0.0)
            signed_qty = qty if side == "BUY" else -qty
            entry_price = _safe_float(trade.entry_price, 0.0)
            mark_price = entry_price
            try:
                mark_price = _safe_float(fetcher.get_ticker(trade.pair).get("price"), entry_price)
            except Exception:
                pass
            unrealized = (mark_price - entry_price) * signed_qty
            rows.append(
                {
                    "trade_id": trade.trade_id,
                    "symbol": trade.pair,
                    "side": "LONG" if signed_qty >= 0 else "SHORT",
                    "quantity": signed_qty,
                    "entry_price": entry_price,
                    "avg_entry_price": entry_price,
                    "mark_price": mark_price,
                    "notional": abs(signed_qty * mark_price),
                    "unrealized_pnl": _safe_float(unrealized, 0.0),
                }
            )
        if target_symbol:
            return [row for row in rows if str(row.get("symbol", "")).upper() == target_symbol]
        return rows
    finally:
        session.close()


@app.get("/api/performance")
def get_performance():
    status_model = get_status()
    if hasattr(status_model, "model_dump"):
        status_payload = status_model.model_dump()
    else:
        status_payload = status_model.dict()
    stats_payload = get_stats()
    positions_payload = get_positions()

    mode = str(status_payload.get("mode", "PAPER")).upper()
    balance = _safe_float(
        status_payload.get("virtual_balance") if mode == "PAPER" else status_payload.get("balance"),
        0.0,
    )
    realized_pnl = _safe_float(
        status_payload.get("paper_realized_pnl") if mode == "PAPER" else stats_payload.get("total_pnl"),
        0.0,
    )
    unrealized_pnl = _safe_float(
        status_payload.get("paper_unrealized_pnl"),
        sum(_safe_float(p.get("unrealized_pnl"), 0.0) for p in positions_payload),
    )
    net_position_value = sum(
        _safe_float(p.get("mark_price"), 0.0) * _safe_float(p.get("quantity"), 0.0)
        for p in positions_payload
    )
    if mode == "PAPER":
        equity = _safe_float(status_payload.get("paper_equity"), balance + net_position_value)
    else:
        equity = balance + unrealized_pnl
    metrics_payload = get_metrics() or {}

    return {
        "mode": mode,
        "equity": _safe_float(equity, 0.0),
        "balance": _safe_float(balance, 0.0),
        "realized_pnl": _safe_float(realized_pnl, 0.0),
        "unrealized_pnl": _safe_float(unrealized_pnl, 0.0),
        "net_position_value": _safe_float(net_position_value, 0.0),
        "daily_pnl": _safe_float(status_payload.get("daily_pnl"), 0.0),
        "win_rate": _safe_float(stats_payload.get("win_rate"), 0.0),
        "total_trades": _safe_int(stats_payload.get("total_trades"), 0),
        "open_positions": _safe_int(len(positions_payload), 0),
        "threshold_after_scaling": _safe_float(status_payload.get("threshold_after_scaling"), 0.0),
        "sharpe_ratio": _safe_float(metrics_payload.get("sharpe_ratio"), 0.0),
        "sortino_ratio": _safe_float(metrics_payload.get("sortino_ratio"), 0.0),
        "max_drawdown": _safe_float(metrics_payload.get("max_drawdown"), 0.0),
        "profit_factor": _safe_float(metrics_payload.get("profit_factor"), 0.0),
        "average_trade": _safe_float(metrics_payload.get("average_trade"), 0.0),
        "exposure_pct": _safe_float(metrics_payload.get("exposure_pct"), 0.0),
        "rolling_volatility": _safe_float(metrics_payload.get("rolling_volatility"), 0.0),
    }


@app.post("/api/trades/{trade_id}/force_close")
def force_close_trade(trade_id: str):
    monitor = _build_monitor()
    closed_id = monitor.force_close_trade(trade_id)
    if not closed_id:
        raise HTTPException(status_code=404, detail=f"Open trade not found: {trade_id}")
    _apply_post_close_updates(closed_id, monitor=monitor)
    return {"status": "force_closed", "trade_id": closed_id}


@app.post("/reconcile")
def enqueue_reconciliation(symbol: str = ACTIVE_SYMBOL):
    target_symbol = str(symbol or ACTIVE_SYMBOL).strip() or ACTIVE_SYMBOL
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
def get_stats(symbol: Optional[str] = None):
    session = get_session()
    try:
        closed_query = session.query(Trade).filter(Trade.status == "closed")
        closed_query = _apply_trade_symbol_scope(closed_query, symbol=symbol)

        total = closed_query.count()
        wins = closed_query.filter(Trade.pnl > 0).count()
        long_trades = closed_query.filter(Trade.side == "BUY").count()
        short_trades = closed_query.filter(Trade.side == "SELL").count()
        total_pnl = closed_query.with_entities(func.sum(Trade.pnl)).scalar() or 0
        avg_r = closed_query.with_entities(func.avg(Trade.r_multiple)).scalar() or 0

        return {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": total - wins,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "win_rate": wins / total if total > 0 else 0,
            "total_pnl": _safe_float(total_pnl, 0.0),
            "avg_r_multiple": _safe_float(avg_r, 0.0),
        }
    finally:
        session.close()


@app.get("/api/pnl-curve")
def get_pnl_curve(days: int = 30, symbol: Optional[str] = None):
    session = get_session()
    try:
        since = datetime.utcnow() - timedelta(days=days)
        query = session.query(Trade).filter(Trade.status == "closed", Trade.exit_time >= since)
        query = _apply_trade_symbol_scope(query, symbol=symbol)
        trades = query.order_by(Trade.exit_time).all()
        cumulative = 0.0
        curve = []
        for t in trades:
            cumulative += _safe_float(t.pnl, 0.0)
            curve.append(
                {
                    "time": (t.exit_time or t.entry_time or datetime.utcnow()).isoformat(),
                    "pnl": _safe_float(cumulative, 0.0),
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
                "timestamp": (p.timestamp or datetime.utcnow()).isoformat(),
                "pair": p.pair,
                "prob_up": _safe_float(p.prob_up, 0.0),
                "prob_down": _safe_float(p.prob_down, 0.0),
                "expected_move": _safe_float(p.expected_move, 0.0),
                "confidence": _safe_float(p.confidence, 0.0),
                "volatility_regime": p.volatility_regime,
                "market_regime": p.market_regime,
                "acted_on": p.acted_on,
            }
            for p in preds
        ]
    finally:
        session.close()


@app.get("/api/ai/current-score", response_model=AIScoreResponse)
def get_ai_current_score(symbol: Optional[str] = None):
    session = get_session()
    try:
        pred_query = session.query(Prediction)
        target_symbol = _normalized_symbol(symbol) if symbol else (ACTIVE_SYMBOL if _is_symbol_scoped_runtime() else None)
        if target_symbol:
            pred_query = pred_query.filter(Prediction.pair == target_symbol)
        latest_pred = pred_query.order_by(desc(Prediction.timestamp)).first()

        closed_query = session.query(Trade).filter(Trade.status == "closed")
        closed_query = _apply_trade_symbol_scope(closed_query, symbol=symbol)
        total = closed_query.count()
        wins = closed_query.filter(Trade.pnl > 0).count()
        win_rate = wins / total if total > 0 else 0.0

        if latest_pred:
            return AIScoreResponse(
                ai_score=_safe_float(latest_pred.confidence, 0.0),
                confidence=_safe_float(latest_pred.confidence, 0.0),
                model_version=str(latest_pred.model_version or "unknown"),
                win_rate=_safe_float(win_rate, 0.0),
                timestamp=(latest_pred.timestamp or datetime.utcnow()).isoformat(),
            )

        return AIScoreResponse(
            ai_score=0.0,
            confidence=0.0,
            model_version="unknown",
            win_rate=_safe_float(win_rate, 0.0),
            timestamp=None,
        )
    finally:
        session.close()


@app.get("/api/ai/history")
def get_ai_history(limit: int = 180, symbol: Optional[str] = None):
    session = get_session()
    try:
        query = session.query(Prediction)
        target_symbol = _normalized_symbol(symbol) if symbol else (ACTIVE_SYMBOL if _is_symbol_scoped_runtime() else None)
        if target_symbol:
            query = query.filter(Prediction.pair == target_symbol)
        rows = query.order_by(desc(Prediction.timestamp)).limit(limit).all()
        return [
            {
                "timestamp": (p.timestamp or datetime.utcnow()).isoformat(),
                "pair": p.pair,
                "ai_score": _safe_float(p.confidence, 0.0),
                "confidence": _safe_float(p.confidence, 0.0),
                "prob_up": _safe_float(p.prob_up, 0.0),
                "prob_down": _safe_float(p.prob_down, 0.0),
                "expected_move": _safe_float(p.expected_move, 0.0),
                "model_version": p.model_version or "unknown",
            }
            for p in reversed(rows)
        ]
    finally:
        session.close()


@app.get("/api/ai/analytics")
def get_ai_analytics(days: int = 7, symbol: Optional[str] = None):
    lookback_days = max(1, min(int(days), 365))
    since = datetime.utcnow() - timedelta(days=lookback_days)
    session = get_session()
    try:
        trades_query = session.query(Trade).filter(Trade.status == "closed", Trade.exit_time >= since)
        trades_query = _apply_trade_symbol_scope(trades_query, symbol=symbol)
        trades = trades_query.order_by(Trade.exit_time.asc()).all()

        timeline: list[dict[str, Any]] = []
        confidence_values: list[float] = []
        cumulative = 0.0
        peak = 0.0

        for trade in trades:
            pred_payload = trade.prediction or {}
            base_score = _safe_float(
                trade.base_ai_score
                if trade.base_ai_score is not None
                else pred_payload.get("base_ai_score", trade.confidence or 0.0)
            )
            final_score = _safe_float(
                trade.ai_score
                if trade.ai_score is not None
                else pred_payload.get("ai_score", trade.confidence or 0.0)
            )
            confidence = _safe_float(trade.confidence, final_score)
            pnl = _safe_float(trade.pnl, 0.0)
            cumulative += pnl
            peak = max(peak, cumulative)
            drawdown = cumulative - peak
            confidence_values.append(confidence)

            timeline.append(
                {
                    "timestamp": (trade.exit_time or trade.entry_time).isoformat(),
                    "pair": trade.pair,
                    "before_score": _safe_float(base_score, 0.0),
                    "final_score": _safe_float(final_score, 0.0),
                    "confidence": _safe_float(confidence, 0.0),
                    "pnl": _safe_float(pnl, 0.0),
                    "equity_curve": _safe_float(cumulative, 0.0),
                    "drawdown": _safe_float(drawdown, 0.0),
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
                "total_pnl": _safe_float(r.total_pnl, 0.0),
                "total_trades": int(r.total_trades or 0),
                "win_trades": int(r.win_trades or 0),
                "win_rate": _safe_float(r.win_rate, 0.0),
                "sharpe": _safe_float(r.sharpe, 0.0),
                "avg_contribution": _safe_float(r.avg_contribution, 0.0),
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
                "equity": _safe_float(r.equity, 0.0),
                "return": _safe_float(r.return_value, 0.0),
                "sharpe": _safe_float(r.sharpe, 0.0),
                "sortino": _safe_float(r.sortino, 0.0),
                "max_drawdown": _safe_float(r.max_drawdown, 0.0),
                "win_rate": _safe_float(r.win_rate, 0.0),
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
                "tft_weight": _safe_float(r.tft_weight, 0.0),
                "xgb_weight": _safe_float(r.xgb_weight, 0.0),
                "ppo_weight": _safe_float(r.ppo_weight, 0.0),
                "confidence_threshold": _safe_float(r.confidence_threshold, 0.0),
                "risk_per_trade": _safe_float(r.risk_per_trade, 0.0),
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
                "daily_loss": _safe_float(r.daily_loss, 0.0),
                "max_equity": _safe_float(r.max_equity, 0.0),
                "consecutive_losses": int(r.consecutive_losses or 0),
            }
            for r in rows
        ]
    finally:
        session.close()


@app.get("/api/risk-state/current")
def get_risk_state_current():
    """Current risk-control state including pause/kill/safe-mode flags."""
    session = get_session()
    try:
        latest = session.query(RiskState).order_by(desc(RiskState.date)).first()
        return {
            "symbol": ACTIVE_SYMBOL,
            "paused": bool(_load_engine_state_value(session, "paused", False)),
            "killed": bool(_load_engine_state_value(session, "killed", False)),
            "safe_mode": bool(_load_engine_state_value(session, "safe_mode", False)),
            "prop_cooldown_until": _load_engine_state_value(session, "prop_cooldown_until", None),
            "single_symbol_loss_streak": _safe_int(_load_engine_state_value(session, "single_symbol_loss_streak", 0), 0),
            "daily_loss": _safe_float(getattr(latest, "daily_loss", 0.0), 0.0),
            "max_equity": _safe_float(getattr(latest, "max_equity", 0.0), 0.0),
            "trading_enabled": bool(getattr(latest, "trading_enabled", True)) if latest else True,
            "as_of": getattr(latest, "date", None),
        }
    finally:
        session.close()


@app.get("/api/model-info")
def get_model_info():
    session = get_session()
    try:
        active_model_name = _load_engine_state_value(session, "active_model_name", None)
        active = (
            session.query(ModelMetric)
            .filter(ModelMetric.is_active == True)
            .order_by(desc(ModelMetric.trained_at))
            .first()
        )
        if not active:
            latest_pred = (
                session.query(Prediction)
                .order_by(desc(Prediction.timestamp), desc(Prediction.id))
                .first()
            )
            fallback_version = active_model_name or (latest_pred.model_version if latest_pred else None)
            if not fallback_version:
                return {"active_model": None}
            return {
                "active_model": {
                    "version": str(fallback_version),
                    "trained_at": None,
                    "validation_loss": None,
                    "sharpe_ratio": None,
                    "max_drawdown": None,
                    "accuracy": None,
                }
            }
        return {
            "active_model": {
                "version": active_model_name or active.model_version,
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
            request_kill_switch_rebaseline("reset_kill")
            request_prop_risk_rebaseline("reset_kill")
            return {"status": "kill_reset"}

        if req.action == "reset_safe_mode":
            safe = session.query(EngineState).filter(EngineState.key == "safe_mode").first()
            if safe:
                safe.value = {"value": False}
            paused = session.query(EngineState).filter(EngineState.key == "paused").first()
            if paused:
                paused.value = {"value": False}
            killed = session.query(EngineState).filter(EngineState.key == "killed").first()
            if killed:
                killed.value = {"value": False}
            session.commit()
            request_kill_switch_rebaseline("reset_safe_mode")
            request_prop_risk_rebaseline("reset_safe_mode")
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

    initial_balance = _safe_float(settings.trading.paper_initial_balance, 0.0)

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
                        "signals",
                        "equity_history",
                        "decision_events",
                        "learning_metrics",
                        "agent_performance",
                        "daily_stats",
                        "performance_metrics",
                        "risk_state",
                        "strategy_parameters",
                        "governance_logs",
                        "ai_decision_audit",
                        "llm_usage",
                        "metrics",
                        "position_events",
                        "statistics",
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
                    _upsert_engine_state(session, "decision_cycle_count", {"value": 0})
                    _upsert_engine_state(session, "trade_attempt_count", {"value": 0})
                    _upsert_engine_state(session, "trade_rejection_count", {"value": 0})
                    _upsert_engine_state(session, "trade_rejection_reasons", {"value": {}})
                    _upsert_engine_state(session, "last_decision_timestamp", {"value": None})
                    _upsert_engine_state(session, "last_trade_timestamp", {"value": None})
                    _upsert_engine_state(session, "last_cycle_terminal_state", {"value": None})
                    _upsert_engine_state(session, "last_cycle_reason", {"value": None})
                    _upsert_engine_state(
                        session,
                        "kill_switch_rebaseline",
                        {"value": True, "reason": "hard_reset", "timestamp": stamp},
                    )
                    _upsert_engine_state(
                        session,
                        "prop_rebaseline_requested",
                        {"value": True, "reason": "hard_reset", "timestamp": stamp},
                    )

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
    return {"status": "reset_started", "message": "reset started", "initial_balance": initial_balance}


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
def get_decision_events(limit: int = 50, symbol: Optional[str] = None):
    """Return recent decision cycle events (including NO_TRADE cycles)."""
    session = get_session()
    try:
        query = session.query(DecisionEvent)
        if symbol:
            query = query.filter(DecisionEvent.best_pair == _normalized_symbol(symbol))
        elif _is_symbol_scoped_runtime():
            query = query.filter(
                (DecisionEvent.best_pair == ACTIVE_SYMBOL) | (DecisionEvent.best_pair.is_(None))
            )
        rows = query.order_by(desc(DecisionEvent.timestamp)).limit(max(1, min(int(limit), 500))).all()
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


@app.get("/api/engine-state")
def get_engine_state():
    """Return engine runtime state including model status and fallback mode."""
    session = get_session()
    try:
        engine_running = bool(_load_engine_state_value(session, "running", False))
        tft_disabled = bool(_load_engine_state_value(session, "tft_disabled", False))
        device = str(_load_engine_state_value(session, "device", "unknown"))
        decision_cycle_count = _safe_int(_load_engine_state_value(session, "decision_cycle_count", 0), 0)
        trade_attempt_count = _safe_int(_load_engine_state_value(session, "trade_attempt_count", 0), 0)
        trade_rejection_count = _safe_int(_load_engine_state_value(session, "trade_rejection_count", 0), 0)
        last_decision_timestamp = _load_engine_state_value(session, "last_decision_timestamp", None)
        last_trade_timestamp = _load_engine_state_value(session, "last_trade_timestamp", None)
        last_cycle_terminal_state = _load_engine_state_value(session, "last_cycle_terminal_state", None)
        last_cycle_reason = _load_engine_state_value(session, "last_cycle_reason", None)
        active_model_name = _load_engine_state_value(session, "active_model_name", None)
        rejection_reasons = _load_engine_state_value(session, "trade_rejection_reasons", {}) or {}

        return {
            "symbol": ACTIVE_SYMBOL,
            "engine_running": engine_running,
            "tft_disabled": tft_disabled,
            "device": device,
            "mode": settings.trading.trading_mode.lower(),
            "trading_enabled": settings.trading.trading_enabled,
            "decision_cycle_count": decision_cycle_count,
            "trade_attempt_count": trade_attempt_count,
            "trade_rejection_count": trade_rejection_count,
            "trade_rejection_reasons": rejection_reasons,
            "last_decision_timestamp": last_decision_timestamp,
            "last_trade_timestamp": last_trade_timestamp,
            "last_cycle_terminal_state": last_cycle_terminal_state,
            "last_cycle_reason": last_cycle_reason,
            "active_model_name": active_model_name,
        }
    finally:
        session.close()


@app.get("/api/observability")
def get_observability():
    """Runtime observability payload for dashboards and external monitoring."""
    session = get_session()
    try:
        rejection_reasons = _load_engine_state_value(session, "trade_rejection_reasons", {}) or {}
        last_decision_timestamp = _load_engine_state_value(session, "last_decision_timestamp", None)
        last_trade_timestamp = _load_engine_state_value(session, "last_trade_timestamp", None)
        active_model_name = _load_engine_state_value(session, "active_model_name", None)
        if not active_model_name:
            latest_pred = (
                session.query(Prediction)
                .order_by(desc(Prediction.timestamp), desc(Prediction.id))
                .first()
            )
            active_model_name = str(latest_pred.model_version) if latest_pred and latest_pred.model_version else "unknown"

        return {
            "symbol": ACTIVE_SYMBOL,
            "decision_cycle_count": _safe_int(_load_engine_state_value(session, "decision_cycle_count", 0), 0),
            "trade_attempt_count": _safe_int(_load_engine_state_value(session, "trade_attempt_count", 0), 0),
            "trade_rejection_count": _safe_int(_load_engine_state_value(session, "trade_rejection_count", 0), 0),
            "trade_rejection_reasons": rejection_reasons,
            "last_decision_timestamp": last_decision_timestamp,
            "last_trade_timestamp": last_trade_timestamp,
            "last_cycle_terminal_state": _load_engine_state_value(session, "last_cycle_terminal_state", None),
            "last_cycle_reason": _load_engine_state_value(session, "last_cycle_reason", None),
            "active_model_name": active_model_name,
            "engine_running": bool(_load_engine_state_value(session, "running", False)),
            "paused": bool(_load_engine_state_value(session, "paused", False)),
            "killed": bool(_load_engine_state_value(session, "killed", False)),
            "safe_mode": bool(_load_engine_state_value(session, "safe_mode", False)),
        }
    finally:
        session.close()


@app.get("/api/prometheus")
def get_prometheus_metrics():
    """Prometheus-compatible metrics for runtime counters."""
    session = get_session()
    try:
        decision_cycles = _safe_int(_load_engine_state_value(session, "decision_cycle_count", 0), 0)
        trade_attempts = _safe_int(_load_engine_state_value(session, "trade_attempt_count", 0), 0)
        trade_rejections = _safe_int(_load_engine_state_value(session, "trade_rejection_count", 0), 0)
        rejection_reasons = _load_engine_state_value(session, "trade_rejection_reasons", {}) or {}
        last_decision_timestamp = _load_engine_state_value(session, "last_decision_timestamp", None)
        last_trade_timestamp = _load_engine_state_value(session, "last_trade_timestamp", None)
        paused = 1 if bool(_load_engine_state_value(session, "paused", False)) else 0
        killed = 1 if bool(_load_engine_state_value(session, "killed", False)) else 0
        safe_mode = 1 if bool(_load_engine_state_value(session, "safe_mode", False)) else 0
    finally:
        session.close()

    def _to_unix(ts_value: Any) -> float:
        try:
            return float(datetime.fromisoformat(str(ts_value)).timestamp())
        except Exception:
            return 0.0

    lines = [
        "# TYPE tft_decision_cycles_total counter",
        f'tft_decision_cycles_total{{symbol="{ACTIVE_SYMBOL}"}} {decision_cycles}',
        "# TYPE tft_trade_attempts_total counter",
        f'tft_trade_attempts_total{{symbol="{ACTIVE_SYMBOL}"}} {trade_attempts}',
        "# TYPE tft_trade_rejections_total counter",
        f'tft_trade_rejections_total{{symbol="{ACTIVE_SYMBOL}"}} {trade_rejections}',
        "# TYPE tft_engine_paused gauge",
        f'tft_engine_paused{{symbol="{ACTIVE_SYMBOL}"}} {paused}',
        "# TYPE tft_engine_killed gauge",
        f'tft_engine_killed{{symbol="{ACTIVE_SYMBOL}"}} {killed}',
        "# TYPE tft_engine_safe_mode gauge",
        f'tft_engine_safe_mode{{symbol="{ACTIVE_SYMBOL}"}} {safe_mode}',
        "# TYPE tft_last_decision_unix gauge",
        f'tft_last_decision_unix{{symbol="{ACTIVE_SYMBOL}"}} {_to_unix(last_decision_timestamp)}',
        "# TYPE tft_last_trade_unix gauge",
        f'tft_last_trade_unix{{symbol="{ACTIVE_SYMBOL}"}} {_to_unix(last_trade_timestamp)}',
    ]
    for reason, count in sorted((rejection_reasons or {}).items()):
        reason_label = str(reason).replace('"', "'")
        lines.append(
            f'tft_trade_rejections_reason_total{{symbol="{ACTIVE_SYMBOL}",reason="{reason_label}"}} {_safe_int(count, 0)}'
        )
    content = "\n".join(lines) + "\n"
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/api/metrics")
def get_metrics():
    """Return institutional performance metrics for this engine's symbol."""
    from engine.metrics import PerformanceTracker
    target_symbol = "PORTFOLIO" if _quant_engine_enabled() else ACTIVE_SYMBOL
    session = get_session()
    try:
        latest = (
            session.query(MetricSnapshot)
            .filter(MetricSnapshot.symbol == target_symbol)
            .order_by(MetricSnapshot.timestamp.desc(), MetricSnapshot.id.desc())
            .first()
        )
        if latest is not None:
            return _json_safe(
                {
                    "symbol": target_symbol,
                    "timestamp": latest.timestamp.isoformat() if latest.timestamp else datetime.utcnow().isoformat(),
                    "sharpe_ratio": _safe_float(latest.sharpe, 0.0),
                    "sortino_ratio": _safe_float(latest.sortino, 0.0),
                    "max_drawdown": _safe_float(latest.max_drawdown, 0.0),
                    "win_rate": _safe_float(latest.win_rate, 0.0),
                    "profit_factor": _safe_float(latest.profit_factor, 0.0),
                    "average_trade": _safe_float(latest.average_trade, 0.0),
                    "exposure_pct": _safe_float(latest.exposure, 0.0),
                    "equity": _safe_float(latest.equity, 0.0),
                    "rolling_volatility": _safe_float(latest.rolling_volatility, 0.0),
                    "total_trades": _safe_int(latest.total_trades, 0),
                    "win_count": _safe_int(latest.winning_trades, 0),
                    "loss_count": _safe_int(latest.losing_trades, 0),
                }
            )
    finally:
        session.close()

    if _quant_engine_enabled():
        from engine.metrics import PortfolioMetrics
        session = get_session()
        try:
            symbols = sorted(
                {
                    str(row[0])
                    for row in session.query(Trade.pair).distinct().all()
                    if row and row[0]
                }
            )
        finally:
            session.close()
        payload = PortfolioMetrics(symbols=symbols or [ACTIVE_SYMBOL]).compute_portfolio_metrics()
        return _json_safe(
            {
                "symbol": target_symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "sharpe_ratio": _safe_float(payload.get("portfolio_sharpe"), 0.0),
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_trade": 0.0,
                "exposure_pct": 0.0,
                "equity": 0.0,
                "rolling_volatility": _safe_float(payload.get("rolling_volatility"), 0.0),
                "total_trades": 0,
                "win_count": 0,
                "loss_count": 0,
            }
        )
    tracker = PerformanceTracker(ACTIVE_SYMBOL)
    return _json_safe(tracker.compute_metrics())


@app.get("/api/equity")
def get_equity(limit: int = 300, bucket_minutes: int = 1, symbol: Optional[str] = None):
    target_symbol = _normalized_symbol(symbol) if symbol else ("PORTFOLIO" if _quant_engine_enabled() else ACTIVE_SYMBOL)
    bucket = max(1, min(int(bucket_minutes), 60))
    session = get_session()
    try:
        rows = (
            session.query(EquityHistory)
            .filter(EquityHistory.symbol == target_symbol)
            .order_by(EquityHistory.timestamp.desc(), EquityHistory.id.desc())
            .limit(max(10, min(int(limit), 5000)))
            .all()
        )
        if rows:
            bucketed: Dict[str, Dict[str, Any]] = {}
            for r in reversed(rows):
                ts = r.timestamp or datetime.utcnow()
                minute_slot = ts.replace(second=0, microsecond=0)
                if bucket > 1:
                    minute_floor = minute_slot.minute - (minute_slot.minute % bucket)
                    minute_slot = minute_slot.replace(minute=minute_floor)
                bucketed[minute_slot.isoformat()] = {
                    "timestamp": minute_slot.isoformat(),
                    "equity": _safe_float(r.equity, 0.0),
                    "balance": _safe_float(r.balance, 0.0),
                    "realized_pnl": _safe_float(r.realized_pnl, 0.0),
                    "unrealized_pnl": _safe_float(r.unrealized_pnl, 0.0),
                    "exposure": _safe_float(r.exposure, 0.0),
                    "open_positions": _safe_int(r.open_positions, 0),
                }
            return list(bucketed.values())

        performance = get_performance()
        return [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "equity": _safe_float(performance.get("equity"), 0.0),
                "balance": _safe_float(performance.get("balance"), 0.0),
                "realized_pnl": _safe_float(performance.get("realized_pnl"), 0.0),
                "unrealized_pnl": _safe_float(performance.get("unrealized_pnl"), 0.0),
                "exposure": 0.0,
                "open_positions": _safe_int(performance.get("open_positions"), 0),
            }
        ]
    finally:
        session.close()


@app.get("/api/metrics/portfolio")
def get_portfolio_metrics():
    """Return portfolio-level metrics across all symbols."""
    from engine.metrics import PortfolioMetrics
    session = get_session()
    try:
        symbols = sorted(
            {
                str(row[0])
                for row in session.query(Trade.pair).distinct().all()
                if row and row[0]
            }
        )
        if _is_symbol_scoped_runtime():
            symbols = [ACTIVE_SYMBOL]
    finally:
        session.close()
    portfolio = PortfolioMetrics(symbols=symbols or [ACTIVE_SYMBOL])
    return _json_safe(portfolio.compute_portfolio_metrics())


@app.get("/api/portfolio/aggregate")
def get_portfolio_aggregate():
    """Aggregate portfolio stats by pair from this datastore."""
    session = get_session()
    try:
        closed_query = session.query(Trade).filter(Trade.status == "closed")
        open_query = session.query(Trade).filter(Trade.status == "open")
        if _is_symbol_scoped_runtime():
            closed_query = closed_query.filter(Trade.pair == ACTIVE_SYMBOL)
            open_query = open_query.filter(Trade.pair == ACTIVE_SYMBOL)

        closed_rows = closed_query.all()
        open_rows = open_query.all()

        by_symbol: Dict[str, Dict[str, Any]] = {}
        for trade in closed_rows:
            symbol = str(trade.pair or ACTIVE_SYMBOL)
            bucket = by_symbol.setdefault(
                symbol,
                {"symbol": symbol, "closed_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "total_pnl": 0.0, "open_trades": 0},
            )
            bucket["closed_trades"] += 1
            pnl_value = _safe_float(trade.pnl, 0.0)
            bucket["total_pnl"] += pnl_value
            if pnl_value > 0:
                bucket["wins"] += 1
            else:
                bucket["losses"] += 1

        for trade in open_rows:
            symbol = str(trade.pair or ACTIVE_SYMBOL)
            bucket = by_symbol.setdefault(
                symbol,
                {"symbol": symbol, "closed_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "total_pnl": 0.0, "open_trades": 0},
            )
            bucket["open_trades"] += 1

        for payload in by_symbol.values():
            total = int(payload["closed_trades"])
            payload["win_rate"] = (float(payload["wins"]) / total) if total > 0 else 0.0
            payload["total_pnl"] = _safe_float(payload["total_pnl"], 0.0)

        symbols = sorted(by_symbol.keys())
        totals = {
            "symbols": len(symbols),
            "closed_trades": sum(int(v["closed_trades"]) for v in by_symbol.values()),
            "open_trades": sum(int(v["open_trades"]) for v in by_symbol.values()),
            "wins": sum(int(v["wins"]) for v in by_symbol.values()),
            "losses": sum(int(v["losses"]) for v in by_symbol.values()),
            "total_pnl": _safe_float(sum(_safe_float(v["total_pnl"], 0.0) for v in by_symbol.values()), 0.0),
        }
        totals["win_rate"] = (
            float(totals["wins"]) / float(totals["closed_trades"])
            if int(totals["closed_trades"]) > 0
            else 0.0
        )
        return {"symbols": symbols, "totals": totals, "by_symbol": [by_symbol[sym] for sym in symbols]}
    finally:
        session.close()


@app.get("/api/universe")
def get_universe():
    session = get_session()
    try:
        symbols = sorted(
            {
                str(row[0])
                for row in session.query(Trade.pair).distinct().all()
                if row and row[0]
            }
        )
        if not symbols:
            symbols = sorted(
                {
                    str(row[0])
                    for row in session.query(SignalRecord.pair).distinct().all()
                    if row and row[0]
                }
            )
    finally:
        session.close()

    if not symbols:
        env_universe = [
            item.strip().upper().replace("/", "-")
            for item in str(os.getenv("UNIVERSE", "")).split(",")
            if item.strip()
        ]
        symbols = sorted(set(env_universe))

    if _is_symbol_scoped_runtime():
        return {"symbols": [ACTIVE_SYMBOL]}
    return {"symbols": symbols or [ACTIVE_SYMBOL]}


@app.get("/status", response_model=StatusResponse)
def get_status_compat():
    return get_status()


@app.get("/trades", response_model=List[TradeResponse])
def get_trades_compat(limit: int = 50, status: Optional[str] = None, symbol: Optional[str] = None):
    return get_trades(limit=limit, status=status, symbol=symbol)


@app.get("/positions")
def get_positions_compat(symbol: Optional[str] = None):
    return get_positions(symbol=symbol)


@app.get("/performance")
def get_performance_compat():
    return get_performance()


@app.get("/metrics")
def get_metrics_compat():
    return get_metrics()


@app.get("/equity")
def get_equity_compat(limit: int = 300):
    return get_equity(limit=limit)


def start_api():
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run(app, host=settings.dashboard.api_host, port=settings.dashboard.api_port)


if __name__ == "__main__":
    start_api()
