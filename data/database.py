"""
Database models for trade logs, predictions, model metrics, and state.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, Boolean, JSON,
    create_engine, Index,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

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
    ai_reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_trades_status_entry", "status", "entry_time"),
    )


class Prediction(Base):
    """Forecast log."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    pair = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    prob_up = Column(Float, nullable=False)
    prob_down = Column(Float, nullable=False)
    expected_move = Column(Float, nullable=False)
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


# ---------------------------------------------------------------------------
# Database session management
# ---------------------------------------------------------------------------

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database.url, pool_pre_ping=True, pool_size=10)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def init_db() -> None:
    """Create all tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
