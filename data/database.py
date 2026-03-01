"""
Database models for trade logs, predictions, model metrics, and state.
"""
from __future__ import annotations

import atexit
import importlib
import threading
import uuid
from datetime import datetime
from typing import Optional

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
from loguru import logger
from sqlalchemy.dialects.postgresql import JSONB

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


def _configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    if settings.database.sqlite_wal_mode:
        cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=30000;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA temp_store=MEMORY;")
    cursor.execute("PRAGMA cache_size=100000;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


def _create_sqlite_engine():
    sqlite_url = settings.database.sqlite_url
    engine = create_engine(
        sqlite_url,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False, "timeout": 30},
        pool_size=1,
        max_overflow=0,
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


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine

        if settings.database.database_mode == "SQLITE":
            logger.info(f"Using SQLite database: {settings.database.sqlite_resolved_path}")
            _engine = _create_sqlite_engine()
            _validate_connection(_engine)
            if settings.database.sqlite_wal_mode:
                logger.info("WAL mode enabled")
            logger.info("SQLite optimized for single-node execution")
            return _engine

        logger.info(
            f"Using PostgreSQL database: {settings.database.host}:{settings.database.port}"
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
