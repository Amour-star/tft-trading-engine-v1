"""Paper account reset utilities."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from sqlalchemy import inspect, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from data.database import (
    AgentPerformance,
    AIDecisionAudit,
    DailyStats,
    DecisionEvent,
    EngineState,
    EquityHistory,
    GovernanceLog,
    LearningMetric,
    LLMUsage,
    MetricSnapshot,
    PerformanceMetric,
    Position,
    PositionEvent,
    Prediction,
    RiskState,
    SignalRecord,
    Statistics,
    StrategyParameter,
    Trade,
)
import data.database as database


TABLES_TO_CLEAR: tuple[tuple[str, type[Any]], ...] = (
    ("trades", Trade),
    ("positions", Position),
    ("predictions", Prediction),
    ("signals", SignalRecord),
    ("decision_events", DecisionEvent),
    ("learning_metrics", LearningMetric),
    ("agent_performance", AgentPerformance),
    ("daily_stats", DailyStats),
    ("performance_metrics", PerformanceMetric),
    ("risk_state", RiskState),
    ("strategy_parameters", StrategyParameter),
    ("governance_logs", GovernanceLog),
    ("ai_decision_audit", AIDecisionAudit),
    ("llm_usage", LLMUsage),
    ("equity_history", EquityHistory),
    ("metrics", MetricSnapshot),
    ("position_events", PositionEvent),
    ("statistics", Statistics),
)


@dataclass
class ResetSummary:
    """Aggregated results from a paper account reset."""

    initial_balance: float
    dry_run: bool = True
    deleted: Dict[str, int] = field(default_factory=dict)
    updated: Dict[str, int] = field(default_factory=dict)
    sqlite_deleted: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "initial_balance": self.initial_balance,
            "dry_run": self.dry_run,
            "deleted": self.deleted,
            "updated": self.updated,
            "sqlite": self.sqlite_deleted,
            "timestamp": self.timestamp.isoformat(),
        }


def _close_open_trades(session, summary: ResetSummary, confirm: bool) -> None:
    """Close any trades still marked as open before wiping history."""
    open_count = session.query(Trade).filter(Trade.status == "open").count()
    summary.updated["open_trades_found"] = int(open_count)
    if not confirm or open_count == 0:
        return

    now = datetime.utcnow()
    result = session.execute(
        update(Trade)
        .where(Trade.status == "open")
        .values(
            status="closed",
            exit_time=now,
            exit_price=Trade.entry_price,
            pnl=0.0,
            pnl_pct=0.0,
            r_multiple=0.0,
            exit_reason="reset",
        )
    )
    closed = int(result.rowcount or open_count)
    summary.updated["open_trades_closed"] = closed
    logger.info(
        "Closed {} open paper trade(s) before clearing history", closed
    )


def _collect_table_counts(session, available_tables: set[str], summary: ResetSummary) -> None:
    """Gather row counts for each table we intend to clear."""
    for label, model in TABLES_TO_CLEAR:
        table_name = model.__tablename__
        if table_name not in available_tables:
            logger.warning("Skipping missing table '{}' during reset", model.__tablename__)
            summary.deleted[label] = 0
            continue
        count = session.query(model).count()
        summary.deleted[label] = int(count)


def _delete_tables(session, available_tables: set[str], summary: ResetSummary) -> None:
    """Delete all rows from the target tables."""
    for label, model in TABLES_TO_CLEAR:
        table_name = model.__tablename__
        if table_name not in available_tables:
            continue
        result = session.execute(text(f"DELETE FROM {table_name}"))
        rows = int(result.rowcount or 0)
        summary.deleted[label] = rows
        logger.info("Deleted {} rows from {}", rows, label)


def _sqlite_table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Check if a table exists in the paper SQLite database."""
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    )
    return cursor.fetchone() is not None


def _collect_paper_sqlite_counts() -> Dict[str, int]:
    """Return the number of rows currently in the paper sqlite tables."""
    counts: Dict[str, int] = {"state": 0, "positions": 0, "trades": 0}
    db_path = Path(settings.trading.paper_db_path)
    if not db_path.exists():
        return counts

    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    try:
        for table in counts.keys():
            if not _sqlite_table_exists(conn, table):
                continue
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = int(row[0]) if row else 0
    finally:
        conn.close()
    return counts


def _ensure_paper_sqlite_tables(conn: sqlite3.Connection) -> None:
    """Create paper sqlite tables if they are missing."""
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
    cols = {row[1] for row in conn.execute("PRAGMA table_info(positions)").fetchall()}
    if "entry_fee_total" not in cols:
        conn.execute(
            "ALTER TABLE positions ADD COLUMN entry_fee_total REAL NOT NULL DEFAULT 0.0"
        )


def _truncate_paper_sqlite(initial_balance: float, summary: ResetSummary) -> Dict[str, int]:
    """Reset the sqlite paper data to a clean state."""
    db_path = Path(settings.trading.paper_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    try:
        _ensure_paper_sqlite_tables(conn)
        deleted_counts: Dict[str, int] = {}
        for table in ("state", "positions", "trades"):
            if not _sqlite_table_exists(conn, table):
                deleted_counts[table] = 0
                continue
            cursor = conn.execute(f"DELETE FROM {table}")
            deleted_counts[table] = int(cursor.rowcount if cursor.rowcount not in (None, -1) else 0)
        conn.executemany(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
            (("starting_balance", initial_balance), ("balance", initial_balance), ("realized_pnl", 0.0)),
        )
        conn.commit()
        conn.execute("VACUUM")
        logger.info("Reset paper sqlite snapshot at {}", db_path)
        return deleted_counts
    finally:
        conn.close()


def _flag_engine_reset(session, initial_balance: float) -> None:
    """Signal the trading engine to reload executor state on the next cycle."""
    now_iso = datetime.utcnow().isoformat()
    payload = {
        "value": True,
        "initial_balance": initial_balance,
        "timestamp": now_iso,
    }
    state = session.query(EngineState).filter(EngineState.key == "paper_reset_pending").first()
    if state:
        state.value = payload
        state.updated_at = datetime.utcnow()
    else:
        session.add(EngineState(key="paper_reset_pending", value=payload))
    day_payload = {"date": datetime.utcnow().date().isoformat(), "equity": float(initial_balance)}
    baseline_payload = {"equity": float(initial_balance)}
    rebaseline_payload = {"value": True, "reason": "paper_reset", "timestamp": now_iso}
    prop_rebaseline_payload = {"value": True, "reason": "paper_reset", "timestamp": now_iso}

    day_state = session.query(EngineState).filter(EngineState.key == "kill_switch_day_start_equity").first()
    if day_state:
        day_state.value = day_payload
        day_state.updated_at = datetime.utcnow()
    else:
        session.add(EngineState(key="kill_switch_day_start_equity", value=day_payload))

    init_state = session.query(EngineState).filter(EngineState.key == "kill_switch_initial_equity").first()
    if init_state:
        init_state.value = baseline_payload
        init_state.updated_at = datetime.utcnow()
    else:
        session.add(EngineState(key="kill_switch_initial_equity", value=baseline_payload))

    rebaseline_state = session.query(EngineState).filter(EngineState.key == "kill_switch_rebaseline").first()
    if rebaseline_state:
        rebaseline_state.value = rebaseline_payload
        rebaseline_state.updated_at = datetime.utcnow()
    else:
        session.add(EngineState(key="kill_switch_rebaseline", value=rebaseline_payload))

    prop_day_state = session.query(EngineState).filter(EngineState.key == "prop_day_start_equity").first()
    if prop_day_state:
        prop_day_state.value = {"value": day_payload}
        prop_day_state.updated_at = datetime.utcnow()
    else:
        session.add(EngineState(key="prop_day_start_equity", value={"value": day_payload}))

    prop_max_state = session.query(EngineState).filter(EngineState.key == "prop_max_equity").first()
    if prop_max_state:
        prop_max_state.value = {"value": float(initial_balance)}
        prop_max_state.updated_at = datetime.utcnow()
    else:
        session.add(EngineState(key="prop_max_equity", value={"value": float(initial_balance)}))

    prop_streak_state = session.query(EngineState).filter(EngineState.key == "single_symbol_loss_streak").first()
    if prop_streak_state:
        prop_streak_state.value = {"value": 0}
        prop_streak_state.updated_at = datetime.utcnow()
    else:
        session.add(EngineState(key="single_symbol_loss_streak", value={"value": 0}))

    prop_cooldown_state = session.query(EngineState).filter(EngineState.key == "prop_cooldown_until").first()
    if prop_cooldown_state:
        prop_cooldown_state.value = {"value": None}
        prop_cooldown_state.updated_at = datetime.utcnow()
    else:
        session.add(EngineState(key="prop_cooldown_until", value={"value": None}))

    prop_rebaseline_state = session.query(EngineState).filter(EngineState.key == "prop_rebaseline_requested").first()
    if prop_rebaseline_state:
        prop_rebaseline_state.value = prop_rebaseline_payload
        prop_rebaseline_state.updated_at = datetime.utcnow()
    else:
        session.add(EngineState(key="prop_rebaseline_requested", value=prop_rebaseline_payload))
    logger.info("Marked engine state for paper reset reconciliation")


def _is_sqlite_corruption_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "database disk image is malformed" in msg


def reset_paper_account(
    initial_balance: Optional[float] = None, *,
    confirm: bool = False,
) -> ResetSummary:
    """
    Wipe paper account state and metrics.

    Args:
        initial_balance: New starting balance for the paper wallet.
        confirm: Apply the changes. Dry-run otherwise.
    """
    target_balance = float(
        initial_balance if initial_balance is not None else settings.trading.paper_initial_balance
    )
    summary = ResetSummary(initial_balance=target_balance, dry_run=not confirm)
    sqlite_counts = _collect_paper_sqlite_counts()
    summary.sqlite_deleted.update(sqlite_counts)
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        engine = database.get_engine()
        session = sessionmaker(bind=engine)()
        try:
            available_tables = set(inspect(session.connection()).get_table_names())
            _close_open_trades(session, summary, confirm)
            _collect_table_counts(session, available_tables, summary)
            if confirm:
                _delete_tables(session, available_tables, summary)
                deleted_sqlite = _truncate_paper_sqlite(target_balance, summary)
                summary.sqlite_deleted.update(deleted_sqlite)
                _flag_engine_reset(session, target_balance)
                summary.updated["reset_flag_set"] = 1
                summary.dry_run = False
                session.commit()
                try:
                    summary.updated["post_reset_trade_count"] = int(session.query(Trade).count())
                except Exception:
                    summary.updated["post_reset_trade_count"] = -1
            else:
                session.rollback()
            logger.info("Paper reset summary: {}", summary.to_dict())
            return summary
        except SQLAlchemyError as exc:
            session.rollback()
            if (
                settings.database.database_mode == "SQLITE"
                and _is_sqlite_corruption_error(exc)
                and attempt < max_attempts
            ):
                logger.error(
                    "Paper reset detected corrupt SQLite database; rebuilding and retrying (attempt {}/{})",
                    attempt + 1,
                    max_attempts,
                )
                database.dispose_engine()
                database.init_db()
                continue
            logger.exception("Paper reset aborted due to database error: {}", exc)
            raise
        except Exception as exc:
            session.rollback()
            logger.exception("Paper reset failed: {}", exc)
            raise
        finally:
            session.close()

    # loop always returns or raises
    return summary
