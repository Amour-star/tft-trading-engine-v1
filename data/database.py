"""
Database models for trade logs, predictions, model metrics, and state.
"""
from __future__ import annotations

import atexit
import importlib
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    Boolean,
    JSON,
    Index,
    Uuid,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, scoped_session, sessionmaker
from sqlalchemy.pool import NullPool
from loguru import logger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import OperationalError

from config.settings import settings


class Base(DeclarativeBase):
    pass


class Trade(Base):
    """Complete trade record."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(64), unique=True, nullable=False, index=True)
    pair = Column(String(20), nullable=False, index=True)
    side = Column(String(4), nullable=False)  # BUY / SELL
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    r_multiple = Column(Float, nullable=True)
    exit_reason = Column(String(50), nullable=True)  # target, stop, trail, signal, manual, error
    slippage_bps = Column(Float, nullable=True)
    commission = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="open")  # open, closed, error
    model_version = Column(String(50), nullable=True)
    features_at_entry = Column(JSON, nullable=True)
    prediction = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    ai_score = Column(Float, nullable=True)
    base_ai_score = Column(Float, nullable=True)
    tft_score = Column(Float, nullable=True)
    xgb_score = Column(Float, nullable=True)
    ppo_score = Column(Float, nullable=True)
    gov_adjust = Column(Float, nullable=True)
    final_ai_score = Column(Float, nullable=True)
    weight_snapshot_json = Column(Text, nullable=True)
    governance_code = Column(String(64), nullable=True)
    ai_reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_trades_status_entry", "status", "entry_time"),
        CheckConstraint("entry_price > 0", name="ck_trades_entry_price_positive"),
    )


class Position(Base):
    """Current reconciled position state by symbol."""
    __tablename__ = "positions"

    symbol = Column(String(20), primary_key=True)
    quantity = Column(Float, nullable=False, default=0.0)
    avg_entry_price = Column(Float, nullable=False, default=0.0)
    source_mode = Column(String(10), nullable=False, default="LIVE")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Prediction(Base):
    """Forecast log."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    pair = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    prob_up = Column(Float, nullable=False)
    prob_down = Column(Float, nullable=False)
    expected_move = Column(Float, nullable=False, default=0.0, server_default=text("0.0"))
    confidence = Column(Float, nullable=False)
    volatility_regime = Column(String(20), nullable=True)
    market_regime = Column(String(20), nullable=True)
    forecast_vector = Column(JSON, nullable=True)
    model_version = Column(String(50), nullable=True)
    acted_on = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class HistoricalCandle(Base):
    """Persisted historical OHLCV candles for training and audit reproducibility."""
    __tablename__ = "historical_candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(16), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0.0)
    source = Column(String(32), nullable=False, default="fetch_history")
    ingested_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_historical_candles_pair_tf_ts", "pair", "timeframe", "timestamp", unique=True),
    )


class SignalRecord(Base):
    """Multi-factor signal log for audit and analytics."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    pair = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, default="1min")
    decision = Column(String(10), nullable=False, default="HOLD")  # LONG, SHORT, HOLD
    side = Column(String(6), nullable=False, default="HOLD")  # BUY, SELL, HOLD
    signal_score = Column(Float, nullable=False, default=0.0)
    momentum_factor = Column(Float, nullable=False, default=0.0)
    volatility_factor = Column(Float, nullable=False, default=0.0)
    trend_factor = Column(Float, nullable=False, default=0.0)
    mean_reversion_factor = Column(Float, nullable=False, default=0.0)
    volume_factor = Column(Float, nullable=False, default=0.0)
    volume_imbalance = Column(Float, nullable=False, default=0.0)
    model_confidence = Column(Float, nullable=False, default=0.0)
    regime = Column(String(32), nullable=True)
    payload = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_signals_pair_timeframe_ts", "pair", "timeframe", "timestamp"),
    )


class EquityHistory(Base):
    """Time-series account state used for live equity curve and exposure analytics."""
    __tablename__ = "equity_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False, index=True)
    mode = Column(String(10), nullable=False, default="PAPER")
    balance = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    equity = Column(Float, nullable=False, default=0.0)
    exposure = Column(Float, nullable=False, default=0.0)
    open_positions = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_equity_history_symbol_ts", "symbol", "timestamp"),
    )


class ModelMetric(Base):
    """Model performance tracking."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False, index=True)
    trained_at = Column(DateTime, nullable=False)
    training_loss = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision_up = Column(Float, nullable=True)
    recall_up = Column(Float, nullable=True)
    is_active = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    """Version registry for AI models (TFT, XGBoost, PPO)."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(20), nullable=False, index=True)  # tft, xgb_meta, ppo
    version = Column(String(64), nullable=False, index=True)
    path = Column(String(512), nullable=False)
    is_active = Column(Boolean, default=True)
    model_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_model_versions_type_created", "model_type", "created_at"),
    )


class LearningMetric(Base):
    """Self-improvement feedback metrics."""
    __tablename__ = "learning_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(64), nullable=False, index=True)
    forecast_accuracy = Column(Float, nullable=True)
    volatility_misread = Column(Boolean, nullable=True)
    confidence_overestimated = Column(Boolean, nullable=True)
    stop_too_tight = Column(Boolean, nullable=True)
    stop_too_wide = Column(Boolean, nullable=True)
    recommended_adjustments = Column(JSON, nullable=True)
    analysis_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class EngineState(Base):
    """Persistent engine state for crash recovery."""
    __tablename__ = "engine_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DailyStats(Base):
    """Daily aggregated statistics."""
    __tablename__ = "daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    balance_start = Column(Float, nullable=True)
    balance_end = Column(Float, nullable=True)
    consecutive_losses = Column(Integer, default=0)
    circuit_breaker_triggered = Column(Boolean, default=False)


class AgentPerformance(Base):
    """Aggregated attribution and performance by agent."""
    __tablename__ = "agent_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent = Column(String(32), nullable=False, unique=True, index=True)
    total_pnl = Column(Float, nullable=False, default=0.0)
    total_trades = Column(Integer, nullable=False, default=0)
    win_trades = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=False, default=0.0)
    sharpe = Column(Float, nullable=False, default=0.0)
    avg_contribution = Column(Float, nullable=False, default=0.0)
    updated_at = Column(String(40), nullable=False, default=lambda: datetime.utcnow().isoformat())


class PerformanceMetric(Base):
    """Per-close snapshot of core performance/risk metrics."""
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(40), nullable=False, index=True)
    equity = Column(Float, nullable=False, default=0.0)
    return_value = Column("return", Float, nullable=False, default=0.0)
    sharpe = Column(Float, nullable=False, default=0.0)
    sortino = Column(Float, nullable=False, default=0.0)
    max_drawdown = Column(Float, nullable=False, default=0.0)
    win_rate = Column(Float, nullable=False, default=0.0)


class RiskState(Base):
    """Daily prop-risk state snapshot."""
    __tablename__ = "risk_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(16), nullable=False, unique=True, index=True)
    trading_enabled = Column(Boolean, nullable=False, default=True)
    daily_loss = Column(Float, nullable=False, default=0.0)
    max_equity = Column(Float, nullable=False, default=0.0)
    consecutive_losses = Column(Integer, nullable=False, default=0)


class StrategyParameter(Base):
    """Versioned strategy parameter snapshots."""
    __tablename__ = "strategy_parameters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(40), nullable=False, index=True)
    tft_weight = Column(Float, nullable=False, default=0.4)
    xgb_weight = Column(Float, nullable=False, default=0.4)
    ppo_weight = Column(Float, nullable=False, default=0.2)
    confidence_threshold = Column(Float, nullable=False, default=0.50)
    risk_per_trade = Column(Float, nullable=False, default=0.01)


class ResearchRun(Base):
    """Research batch metadata for automated strategy discovery."""

    __tablename__ = "research_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(16), nullable=False, default="15min")
    mode = Column(String(10), nullable=False, default="PAPER")
    status = Column(String(20), nullable=False, default="running")
    candidate_count = Column(Integer, nullable=False, default=0)
    accepted_count = Column(Integer, nullable=False, default=0)
    selected_count = Column(Integer, nullable=False, default=0)
    deployed_count = Column(Integer, nullable=False, default=0)
    notes_json = Column(JSON, nullable=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class ResearchStrategy(Base):
    """Persisted strategy candidate definition and evaluation payloads."""

    __tablename__ = "research_strategies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(96), nullable=False, unique=True, index=True)
    run_id = Column(String(64), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(16), nullable=False, default="15min")
    status = Column(String(20), nullable=False, default="rejected")
    score = Column(Float, nullable=False, default=0.0)
    rank_percentile = Column(Float, nullable=False, default=1.0)
    selected = Column(Boolean, nullable=False, default=False)
    deployed = Column(Boolean, nullable=False, default=False)
    failure_reason = Column(Text, nullable=True)
    definition_json = Column(JSON, nullable=False)
    indicators_json = Column(JSON, nullable=True)
    train_metrics_json = Column(JSON, nullable=True)
    test_metrics_json = Column(JSON, nullable=True)
    walk_forward_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_research_strategies_symbol_score", "symbol", "score"),
    )


class ResearchDeployment(Base):
    """Active paper deployments sourced from research runs."""

    __tablename__ = "research_deployments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, index=True)
    strategy_id = Column(String(96), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(16), nullable=False, default="15min")
    deployment_mode = Column(String(10), nullable=False, default="PAPER")
    rank_percentile = Column(Float, nullable=False, default=1.0)
    score = Column(Float, nullable=False, default=0.0)
    is_active = Column(Boolean, nullable=False, default=True)
    deployed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    definition_json = Column(JSON, nullable=False)

    __table_args__ = (
        Index("ix_research_deployments_active", "symbol", "timeframe", "is_active"),
    )


class MetricSnapshot(Base):
    """Institutional metric snapshots for dashboard/api consumers."""
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False, index=True)
    sharpe = Column(Float, nullable=False, default=0.0)
    sortino = Column(Float, nullable=False, default=0.0)
    max_drawdown = Column(Float, nullable=False, default=0.0)
    win_rate = Column(Float, nullable=False, default=0.0)
    profit_factor = Column(Float, nullable=False, default=0.0)
    average_trade = Column(Float, nullable=False, default=0.0)
    exposure = Column(Float, nullable=False, default=0.0)
    equity = Column(Float, nullable=False, default=0.0)
    rolling_volatility = Column(Float, nullable=False, default=0.0)
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)


class GovernanceLog(Base):
    """Raw LLM governance request/response log."""
    __tablename__ = "governance_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    pair = Column(String(20), nullable=False, index=True)
    volatility_regime = Column(String(20), nullable=True)
    rsi_bucket = Column(String(20), nullable=True)
    request_payload = Column(JSON, nullable=False)
    response_payload = Column(JSON, nullable=False)
    approved = Column(Boolean, nullable=False, default=True)
    size_mult = Column(Float, nullable=False, default=1.0)
    conf_adj = Column(Float, nullable=False, default=0.0)
    risk_mode = Column(String(20), nullable=False, default="neutral")
    code = Column(String(64), nullable=False)
    latency_ms = Column(Float, nullable=True)
    from_cache = Column(Boolean, nullable=False, default=False)
    fallback_reason = Column(String(64), nullable=True)
    error = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_governance_logs_pair_time", "pair", "timestamp"),
    )


class AIDecisionAudit(Base):
    """Final AI score composition audit for each evaluated signal."""
    __tablename__ = "ai_decision_audit"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    pair = Column(String(20), nullable=False, index=True)
    base_ai_score = Column(Float, nullable=False)
    ppo_size_mult = Column(Float, nullable=False)
    governance_size_mult = Column(Float, nullable=False)
    governance_conf_adj = Column(Float, nullable=False)
    final_ai_score = Column(Float, nullable=False)
    approved = Column(Boolean, nullable=False, default=True)
    governance_code = Column(String(64), nullable=False)
    volatility_regime = Column(String(20), nullable=True)
    rsi_bucket = Column(String(20), nullable=True)
    metadata_json = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_ai_decision_audit_pair_time", "pair", "timestamp"),
    )


class LLMUsage(Base):
    """Token/cost accounting for governance LLM usage."""
    __tablename__ = "llm_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    provider = Column(String(50), nullable=False)
    model = Column(String(120), nullable=False)
    pair = Column(String(20), nullable=True)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    estimated_cost_usd = Column(Float, nullable=False, default=0.0)
    actual_cost_usd = Column(Float, nullable=False, default=0.0)
    status = Column(String(20), nullable=False)  # success, timeout, error, blocked, invalid_json
    error = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_llm_usage_created", "created_at"),
    )


class DecisionEvent(Base):
    """Cycle-level decision log — persisted even when no trade is opened."""
    __tablename__ = "decision_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    mode = Column(String(10), nullable=False, default="PAPER")
    status = Column(String(20), nullable=False, default="no_trade")  # no_trade, trade_opened
    reason = Column(String(100), nullable=True)
    candidates_evaluated = Column(Integer, nullable=False, default=0)
    candidates_valid = Column(Integer, nullable=False, default=0)
    best_pair = Column(String(20), nullable=True)
    best_score = Column(Float, nullable=True)
    best_ai_score = Column(Float, nullable=True)
    best_confidence = Column(Float, nullable=True)
    best_prob_up = Column(Float, nullable=True)
    best_prob_down = Column(Float, nullable=True)
    regime = Column(String(20), nullable=True)
    volatility_regime = Column(String(20), nullable=True)
    adaptive_threshold = Column(Float, nullable=True)
    top_candidates_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PositionEvent(Base):
    """Event stream for stateless reconciliation requests/results."""
    __tablename__ = "position_events"

    id = Column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    event_type = Column(String(32), nullable=False, index=True)
    details = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False, default=dict)

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('reconcile_request','reconcile_complete','reconcile_failed')",
            name="ck_position_events_event_type",
        ),
        Index("ix_position_events_symbol_time", "symbol", "timestamp"),
    )


class Statistics(Base):
    """Lightweight aggregate statistics that shadow runtime counters."""

    __tablename__ = "statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    total_trades = Column(Integer, nullable=False, default=0)
    wins = Column(Integer, nullable=False, default=0)
    losses = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=False, default=0.0)
    avg_r = Column(Float, nullable=False, default=0.0)
    total_pnl = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# Database session management
# ---------------------------------------------------------------------------

_engine = None
_SessionFactory = None
_SessionLocal = None
_engine_lock = threading.Lock()
_db_runtime_info: dict[str, object] = {
    "backend": None,
    "sqlite_path": None,
    "sqlite_parent": None,
    "sqlite_exists": None,
    "sqlite_parent_writable": None,
    "sqlite_file_writable": None,
    "sqlite_requested_journal_mode": None,
    "sqlite_effective_journal_mode": None,
}


def _update_db_runtime_info(**values: object) -> None:
    _db_runtime_info.update(values)


def get_database_runtime_info() -> dict[str, object]:
    return dict(_db_runtime_info)


def _sqlite_storage_summary(db_path: Path) -> dict[str, object]:
    parent = db_path.parent
    file_exists = db_path.exists()
    return {
        "sqlite_path": str(db_path),
        "sqlite_parent": str(parent),
        "sqlite_exists": file_exists,
        "sqlite_parent_writable": bool(os.access(parent, os.W_OK)),
        "sqlite_file_writable": bool(os.access(db_path, os.W_OK)) if file_exists else True,
    }


def _ensure_sqlite_path_ready(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        db_path.touch(exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            f"SQLite path is not writable: {db_path} (parent={db_path.parent})"
        ) from exc

    storage = _sqlite_storage_summary(db_path)
    _update_db_runtime_info(backend="sqlite", **storage)
    if not bool(storage["sqlite_parent_writable"]):
        raise RuntimeError(f"SQLite parent directory is not writable: {db_path.parent}")
    if not bool(storage["sqlite_file_writable"]):
        raise RuntimeError(f"SQLite database file is not writable: {db_path}")


def _set_sqlite_journal_mode(cursor, db_path: Path) -> str:
    requested_mode = settings.database.sqlite_journal_mode
    fallback_mode = settings.database.sqlite_fallback_journal_mode

    def _apply(mode: str) -> str:
        cursor.execute(f"PRAGMA journal_mode={mode};")
        row = cursor.fetchone()
        effective_mode = str(row[0]).strip().upper() if row and row[0] is not None else mode
        _update_db_runtime_info(
            sqlite_requested_journal_mode=requested_mode,
            sqlite_effective_journal_mode=effective_mode,
        )
        return effective_mode

    try:
        return _apply(requested_mode)
    except sqlite3.Error as exc:
        if fallback_mode and fallback_mode != requested_mode:
            logger.warning(
                "SQLite journal mode {} failed for {} ({}). Falling back to {}.",
                requested_mode,
                db_path,
                exc,
                fallback_mode,
            )
            return _apply(fallback_mode)
        raise RuntimeError(
            "SQLite journal mode setup failed for "
            f"{db_path} (requested={requested_mode}, fallback={fallback_mode or 'none'}): {exc}"
        ) from exc


def _configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    db_path = settings.database.sqlite_resolved_path
    try:
        effective_mode = _set_sqlite_journal_mode(cursor, db_path)
        cursor.execute("PRAGMA busy_timeout=30000;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA temp_store=MEMORY;")
        cursor.execute("PRAGMA cache_size=100000;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        logger.debug(
            "SQLite connection configured: path={} requested_journal_mode={} effective_journal_mode={}",
            db_path,
            settings.database.sqlite_journal_mode,
            effective_mode,
        )
    finally:
        cursor.close()


def _create_sqlite_engine():
    sqlite_url = settings.database.sqlite_url
    engine = create_engine(
        sqlite_url,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=NullPool,
    )
    event.listen(engine, "connect", _configure_sqlite_connection)
    return engine


def _ensure_psycopg2_installed() -> None:
    try:
        importlib.import_module("psycopg2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DATABASE_MODE=POSTGRES requires psycopg2. Install psycopg2-binary or psycopg2."
        ) from exc


def _create_postgres_engine():
    _ensure_psycopg2_installed()
    return create_engine(
        settings.database.postgres_url,
        pool_pre_ping=True,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
    )


def _validate_connection(engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def _validate_sqlite_integrity(engine) -> None:
    """Run a lightweight integrity check before exposing a SQLite engine."""
    with engine.connect() as conn:
        row = conn.execute(text("PRAGMA quick_check")).fetchone()
    status = str(row[0]).strip().lower() if row and row[0] is not None else "ok"
    if status != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {status}")


def _is_sqlite_corruption_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "database disk image is malformed" in message
        or "sqlite integrity check failed" in message
    )


def _quarantine_corrupt_sqlite_files(db_path: Path) -> None:
    """Move corrupt sqlite files away so a clean DB can be recreated."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    candidates = [db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]
    for src in candidates:
        if not src.exists():
            continue
        backup = src.with_name(f"{src.name}.corrupt.{timestamp}")
        try:
            os.replace(src, backup)
            logger.warning("Quarantined corrupt sqlite file: {} -> {}", src, backup)
        except Exception as move_exc:
            logger.warning("Failed to quarantine sqlite file {}: {}", src, move_exc)
            try:
                src.unlink(missing_ok=True)
                logger.warning("Deleted sqlite file after failed quarantine: {}", src)
            except Exception as unlink_exc:
                logger.error("Could not remove corrupt sqlite file {}: {}", src, unlink_exc)


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine

        if settings.database.database_mode == "SQLITE":
            sqlite_path = settings.database.sqlite_resolved_path
            _ensure_sqlite_path_ready(sqlite_path)
            storage = _sqlite_storage_summary(sqlite_path)
            logger.info(
                "Using SQLite database: {} | journal_mode={} | fallback_journal_mode={} | "
                "exists={} | parent_writable={} | file_writable={}",
                sqlite_path,
                settings.database.sqlite_journal_mode,
                settings.database.sqlite_fallback_journal_mode or "none",
                storage["sqlite_exists"],
                storage["sqlite_parent_writable"],
                storage["sqlite_file_writable"],
            )
            recovered = False
            while True:
                _engine = _create_sqlite_engine()
                max_attempts = 8
                try:
                    for attempt in range(1, max_attempts + 1):
                        try:
                            _validate_connection(_engine)
                            break
                        except OperationalError as exc:
                            message = str(exc).lower()
                            if "unable to open database file" not in message or attempt >= max_attempts:
                                raise
                            sleep_s = min(2.0, 0.25 * attempt)
                            logger.warning(
                                "SQLite open failed (attempt {}/{}): {}. Retrying in {:.2f}s",
                                attempt,
                                max_attempts,
                                exc,
                                sleep_s,
                            )
                            time.sleep(sleep_s)
                    _validate_sqlite_integrity(_engine)
                    break
                except Exception as exc:
                    if not recovered and _is_sqlite_corruption_error(exc):
                        logger.error(
                            "SQLite corruption detected at {} ({}). Rebuilding database file.",
                            sqlite_path,
                            exc,
                        )
                        try:
                            _engine.dispose()
                        except Exception:
                            pass
                        _engine = None
                        _quarantine_corrupt_sqlite_files(sqlite_path)
                        recovered = True
                        continue
                    raise
            logger.info(
                "SQLite ready: path={} | effective_journal_mode={} | backend=sqlite",
                sqlite_path,
                get_database_runtime_info().get("sqlite_effective_journal_mode")
                or settings.database.sqlite_journal_mode,
            )
            return _engine

        logger.info(
            f"Using PostgreSQL database: {settings.database.host}:{settings.database.port}"
        )
        _update_db_runtime_info(
            backend="postgres",
            sqlite_path=None,
            sqlite_parent=None,
            sqlite_exists=None,
            sqlite_parent_writable=None,
            sqlite_file_writable=None,
            sqlite_requested_journal_mode=None,
            sqlite_effective_journal_mode=None,
        )
        try:
            _engine = _create_postgres_engine()
            _validate_connection(_engine)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("PostgreSQL unavailable. Fix configuration.") from exc

    return _engine


def get_session() -> Session:
    global _SessionFactory, _SessionLocal
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
        )
    if _SessionLocal is None:
        _SessionLocal = scoped_session(_SessionFactory)
    return _SessionLocal()


def remove_session() -> None:
    if _SessionLocal is not None:
        _SessionLocal.remove()


def run_migrations() -> None:
    """
    Apply schema migrations for the active backend.

    SQLite is migrated in-process via metadata create_all.
    PostgreSQL migrations are intentionally not auto-run in-process.
    """
    if settings.database.database_mode == "SQLITE":
        engine = get_engine()
        Base.metadata.create_all(engine)
        _ensure_sqlite_compat_columns(engine)
        return
    logger.info(
        "DATABASE_MODE=POSTGRES selected. Automatic migrations are disabled; "
        "run PostgreSQL migrations explicitly."
    )


def _ensure_sqlite_compat_columns(engine) -> None:
    """
    Ensure new columns exist on existing SQLite deployments.
    create_all() does not alter existing tables.
    """
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(trades)")).fetchall()
        existing = {str(row[1]) for row in rows}
        add_if_missing = {
            "ai_score": "ai_score FLOAT",
            "base_ai_score": "base_ai_score FLOAT",
            "tft_score": "tft_score FLOAT",
            "xgb_score": "xgb_score FLOAT",
            "ppo_score": "ppo_score FLOAT",
            "gov_adjust": "gov_adjust FLOAT",
            "final_ai_score": "final_ai_score FLOAT",
            "weight_snapshot_json": "weight_snapshot_json TEXT",
            "governance_code": "governance_code VARCHAR(64)",
        }
        for column, ddl in add_if_missing.items():
            if column not in existing:
                conn.execute(text(f"ALTER TABLE trades ADD COLUMN {ddl}"))
                logger.info(f"SQLite migration: added trades.{column}")


def init_db() -> None:
    """Initialize schema for the active database mode."""
    run_migrations()


def dispose_engine() -> None:
    """Dispose DB engine and clear session factory."""
    global _engine, _SessionFactory, _SessionLocal
    if _SessionLocal is not None:
        _SessionLocal.remove()
        _SessionLocal = None
    if _SessionFactory is not None:
        _SessionFactory = None
    if _engine is not None:
        _engine.dispose()
        _engine = None


atexit.register(dispose_engine)


def register_model_version(
    model_type: str,
    version: str,
    path: str,
    model_metadata: Optional[dict] = None,
    activate: bool = True,
) -> Optional[ModelVersion]:
    """Register a model artifact version in the central registry."""
    session = get_session()
    try:
        if activate:
            session.query(ModelVersion).filter(ModelVersion.model_type == model_type).update(
                {ModelVersion.is_active: False}
            )
        record = ModelVersion(
            model_type=model_type,
            version=version,
            path=path,
            is_active=activate,
            model_metadata=model_metadata or {},
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    except Exception as exc:
        session.rollback()
        logger.error(f"Failed to register model version {model_type}:{version}: {exc}")
        return None
    finally:
        session.close()


def get_latest_model_version(model_type: str, active_only: bool = True) -> Optional[ModelVersion]:
    """Fetch the newest model version for a given type."""
    session = get_session()
    try:
        query = session.query(ModelVersion).filter(ModelVersion.model_type == model_type)
        if active_only:
            query = query.filter(ModelVersion.is_active.is_(True))
        return query.order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc()).first()
    finally:
        session.close()


def replace_historical_candles(
    pair: str,
    timeframe: str,
    frame: pd.DataFrame,
    *,
    source: str = "fetch_history",
) -> int:
    """Replace persisted OHLCV history for a symbol/timeframe with the provided frame."""
    if frame is None or frame.empty:
        return 0

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Historical frame missing required columns: {sorted(missing)}")

    working = frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=False, errors="coerce")
    working = working.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    if working.empty:
        return 0

    records = [
        {
            "pair": str(pair).strip().upper(),
            "timeframe": str(timeframe).strip(),
            "timestamp": row.timestamp.to_pydatetime() if hasattr(row.timestamp, "to_pydatetime") else row.timestamp,
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
            "source": str(source),
            "ingested_at": datetime.utcnow(),
        }
        for row in working.itertuples(index=False)
    ]

    session = get_session()
    try:
        session.query(HistoricalCandle).filter(
            HistoricalCandle.pair == str(pair).strip().upper(),
            HistoricalCandle.timeframe == str(timeframe).strip(),
        ).delete(synchronize_session=False)
        session.bulk_insert_mappings(HistoricalCandle, records)
        session.commit()
        return len(records)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
