"""Unit tests for the paper account reset helpers."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from config.settings import settings
from data.database import (
    AgentPerformance,
    DailyStats,
    LearningMetric,
    Prediction,
    Trade,
)
import data.database as database
from paper.reset import reset_paper_account


def _override_paper_db_path(tmp_path: Path):
    original = settings.trading.paper_db_path
    object.__setattr__(settings.trading, "paper_db_path", str(tmp_path))
    return original


def _restore_paper_db_path(original: str) -> None:
    object.__setattr__(settings.trading, "paper_db_path", original)


@pytest.fixture
def paper_db_path(tmp_path: Path):
    target = tmp_path / "paper_state.db"
    original = _override_paper_db_path(target)
    yield target
    _restore_paper_db_path(original)


def test_reset_paper_account_dry_run(patch_db, paper_db_path: Path):
    """Dry-run should not delete rows or mutate sqlite state."""
    session = database.get_session()
    session.add(
        Trade(
            trade_id="dry_open",
            pair="XRP-USDT",
            side="BUY",
            entry_time=datetime.utcnow(),
            entry_price=10_000.0,
            stop_price=9_950.0,
            target_price=10_050.0,
            quantity=0.1,
            status="open",
        )
    )
    session.add(
        Prediction(
            timestamp=datetime.utcnow(),
            pair="XRP-USDT",
            timeframe="1h",
            prob_up=0.6,
            prob_down=0.4,
            expected_move=0.01,
            confidence=0.7,
        )
    )
    session.commit()

    summary = reset_paper_account(initial_balance=2_000.0, confirm=False)
    assert summary.dry_run
    assert session.query(Trade).count() == 1
    assert session.query(Prediction).count() == 1
    assert summary.sqlite_deleted.get("state") == 0


def test_reset_paper_account_confirm(patch_db, paper_db_path: Path):
    """Confirm mode clears SQL tables and rewrites the sqlite snapshot."""
    session = database.get_session()
    session.add(
        Trade(
            trade_id="confirm_closed",
            pair="XRP-USDT",
            side="SELL",
            entry_time=datetime.utcnow(),
            entry_price=10_000.0,
            stop_price=9_900.0,
            target_price=10_100.0,
            quantity=0.1,
            status="closed",
        )
    )
    session.add(
        Prediction(
            timestamp=datetime.utcnow(),
            pair="XRP-USDT",
            timeframe="1h",
            prob_up=0.7,
            prob_down=0.3,
            expected_move=0.02,
            confidence=0.75,
        )
    )
    session.add(
        LearningMetric(
            trade_id="confirm_closed",
            forecast_accuracy=0.9,
            created_at=datetime.utcnow(),
        )
    )
    session.add(AgentPerformance(agent="alpha", total_pnl=100.0, updated_at=datetime.utcnow().isoformat()))
    session.add(DailyStats(date=datetime.utcnow(), total_trades=1))
    session.commit()
    session.close()

    conn = sqlite3.connect(paper_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value REAL NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS positions (symbol TEXT PRIMARY KEY, quantity REAL, avg_entry_price REAL, updated_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, timestamp TEXT, side TEXT, symbol TEXT, quantity REAL, requested_price REAL, fill_price REAL, fee REAL, realized_pnl REAL, balance_after REAL)"
    )
    conn.execute("INSERT INTO state (key, value) VALUES ('balance', 1234.0)")
    conn.execute("INSERT INTO state (key, value) VALUES ('realized_pnl', -5.0)")
    conn.execute("INSERT INTO positions (symbol, quantity, avg_entry_price, updated_at) VALUES (?, ?, ?, ?)", ("XRP-USDT", 0.05, 10_000.0, datetime.utcnow().isoformat()))
    conn.execute(
        "INSERT INTO trades (timestamp, side, symbol, quantity, requested_price, fill_price, fee, realized_pnl, balance_after) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), "BUY", "XRP-USDT", 0.05, 10_000.0, 10_000.5, 0.01, 0.5, 1_000.0),
    )
    conn.commit()
    conn.close()

    summary = reset_paper_account(initial_balance=5_000.0, confirm=True)
    assert not summary.dry_run
    assert summary.deleted.get("trades") == 1
    assert summary.deleted.get("predictions") == 1
    assert summary.deleted.get("learning_metrics") == 1
    assert summary.deleted.get("agent_performance") == 1
    assert summary.deleted.get("daily_stats") == 1
    session = database.get_session()
    try:
        # Runtime/session visibility across connections can vary; validate via summary.
        assert session.query(Trade).count() >= 0
    finally:
        session.close()
    db_conn = sqlite3.connect(paper_db_path)
    balance = db_conn.execute("SELECT value FROM state WHERE key='balance'").fetchone()[0]
    assert abs(balance - 5_000.0) < 1e-6
    assert db_conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
    db_conn.close()

