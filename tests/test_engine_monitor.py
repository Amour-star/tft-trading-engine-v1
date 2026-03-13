from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace


apscheduler_module = ModuleType("apscheduler")
apscheduler_schedulers = ModuleType("apscheduler.schedulers")
apscheduler_background = ModuleType("apscheduler.schedulers.background")
apscheduler_background.BackgroundScheduler = object
apscheduler_schedulers.background = apscheduler_background
apscheduler_module.schedulers = apscheduler_schedulers
sys.modules.setdefault("apscheduler", apscheduler_module)
sys.modules.setdefault("apscheduler.schedulers", apscheduler_schedulers)
sys.modules.setdefault("apscheduler.schedulers.background", apscheduler_background)

docker_module = ModuleType("docker")
docker_errors = ModuleType("docker.errors")
docker_module.DockerClient = object
docker_errors.DockerException = Exception
docker_errors.NotFound = Exception
docker_module.errors = docker_errors
sys.modules.setdefault("docker", docker_module)
sys.modules.setdefault("docker.errors", docker_errors)

from scripts import engine_monitor


def _create_trade_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                entry_time TEXT,
                exit_time TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _container(
    started_at: datetime,
    *,
    status: str = "running",
    health: str = "healthy",
    name: str = "tft-trading-engine-engine-btc-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        status=status,
        reload=lambda: None,
        attrs={
            "State": {
                "StartedAt": started_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "Health": {"Status": health},
            }
        }
    )


def test_no_trade_issue_ignores_fresh_empty_db(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(engine_monitor, "STATE_ROOT", tmp_path)
    monitor = engine_monitor.EngineMonitor.__new__(engine_monitor.EngineMonitor)
    db_path = tmp_path / "btc" / "tft_engine.db"
    _create_trade_db(db_path)

    result = monitor._no_trade_issue(
        "btc",
        _container(datetime.now(timezone.utc) - timedelta(minutes=10)),
    )

    assert result is None
    assert not (tmp_path / "btc" / "trades.db").exists()


def test_no_trade_issue_flags_stale_empty_db(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(engine_monitor, "STATE_ROOT", tmp_path)
    monitor = engine_monitor.EngineMonitor.__new__(engine_monitor.EngineMonitor)
    symbol_dir = tmp_path / "btc"
    db_path = symbol_dir / "tft_engine.db"
    _create_trade_db(db_path)

    stale_dt = datetime.now(timezone.utc) - timedelta(hours=7)
    stale_ts = stale_dt.timestamp()
    os.utime(symbol_dir, (stale_ts, stale_ts))
    os.utime(db_path, (stale_ts, stale_ts))

    result = monitor._no_trade_issue("btc", _container(stale_dt))

    assert result == "No trades for > 6 hours"


def test_check_symbol_ignores_stale_log_matches_when_container_is_healthy(monkeypatch) -> None:
    monitor = engine_monitor.EngineMonitor.__new__(engine_monitor.EngineMonitor)
    container = _container(datetime.now(timezone.utc) - timedelta(minutes=5), health="healthy")

    monkeypatch.setattr(monitor, "_resolve_container", lambda symbol: container)
    monkeypatch.setattr(
        monitor,
        "_get_container_logs",
        lambda container_name, since_dt: "CRITICAL\nModel load failed\nunable to open database file\n",
    )
    monkeypatch.setattr(monitor, "_no_trade_issue", lambda symbol, container=None: None)

    result = monitor._check_symbol("btc")

    assert result.issues == []


def test_check_symbol_flags_recent_sqlite_bootstrap_failures_when_not_healthy(monkeypatch) -> None:
    monitor = engine_monitor.EngineMonitor.__new__(engine_monitor.EngineMonitor)
    container = _container(datetime.now(timezone.utc) - timedelta(minutes=1), health="starting")

    monkeypatch.setattr(monitor, "_resolve_container", lambda symbol: container)
    monkeypatch.setattr(
        monitor,
        "_get_container_logs",
        lambda container_name, since_dt: "sqlite3.OperationalError: unable to open database file",
    )
    monkeypatch.setattr(monitor, "_no_trade_issue", lambda symbol, container=None: None)

    result = monitor._check_symbol("btc")

    assert result.issues == ["log_match:unable to open database file"]


def test_run_once_does_not_restart_for_no_trade_warning(monkeypatch) -> None:
    monitor = engine_monitor.EngineMonitor.__new__(engine_monitor.EngineMonitor)
    monitor.failure_counts = {"btc": 0}
    recorded_events: list[tuple[str, str]] = []
    restarted: list[str] = []

    monkeypatch.setattr(engine_monitor, "ENGINE_NAMES", {"btc": ["tft-trading-engine-engine-btc-1"]})
    monkeypatch.setattr(
        monitor,
        "_check_symbol",
        lambda symbol: engine_monitor.CheckResult(
            symbol="btc",
            container_name="tft-trading-engine-engine-btc-1",
            issues=["No trades for > 6 hours"],
            logs_checked=0,
        ),
    )
    monkeypatch.setattr(monitor, "_restart_container", lambda container_name: restarted.append(container_name))
    monkeypatch.setattr(
        monitor,
        "_write_report",
        lambda event, message: recorded_events.append((event, message)),
    )

    monitor.run_once()

    assert restarted == []
    assert recorded_events[0][0] == "warning_detected"
