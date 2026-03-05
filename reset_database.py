"""Reset trading runtime tables for SQLite/PostgreSQL backends.

Target tables:
  - trades
  - positions
  - metrics
  - equity_history

The command runs in dry-run mode by default. Use --apply to execute.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from data.database import Base

TARGET_TABLES: tuple[str, ...] = ("trades", "positions", "metrics", "equity_history")


def _detect_database_mode(explicit: str) -> str:
    mode = (explicit or "auto").strip().upper()
    if mode in {"SQLITE", "POSTGRES"}:
        return mode
    env_mode = os.getenv("DATABASE_MODE", "").strip().upper()
    if env_mode in {"SQLITE", "POSTGRES"}:
        return env_mode
    return "POSTGRES" if os.getenv("POSTGRES_HOST") else "SQLITE"


def _postgres_url_from_env() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost").strip() or "localhost"
    port = os.getenv("POSTGRES_PORT", "5432").strip() or "5432"
    db = os.getenv("POSTGRES_DB", "tft_trading").strip() or "tft_trading"
    user = os.getenv("POSTGRES_USER", "trader").strip() or "trader"
    password = os.getenv("POSTGRES_PASSWORD", "").strip()
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def _discover_sqlite_targets(explicit_path: str) -> List[Path]:
    targets: List[Path] = []
    if explicit_path:
        targets.append(Path(explicit_path))
    else:
        env_path = os.getenv("SQLITE_PATH", "").strip()
        if env_path:
            if not (os.name == "nt" and env_path.startswith("/")):
                env_candidate = Path(env_path)
                if env_candidate.exists():
                    targets.append(env_candidate)
        else:
            default_candidate = Path("data/tft_engine.db")
            if default_candidate.exists():
                targets.append(default_candidate)

        state_root = Path("state")
        if state_root.exists():
            for db_path in sorted(state_root.glob("*/tft_engine.db")):
                targets.append(db_path)

    resolved: List[Path] = []
    seen = set()
    for path in targets:
        full = path if path.is_absolute() else (Path.cwd() / path)
        full = full.resolve()
        if str(full) in seen:
            continue
        seen.add(str(full))
        resolved.append(full)
    return resolved


def _row_count_or_none(conn, table_name: str) -> int | None:
    try:
        return int(conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar_one())
    except Exception:
        return None


def _reset_tables_in_engine(engine: Engine, apply: bool) -> Dict[str, Dict[str, int | None]]:
    summary: Dict[str, Dict[str, int | None]] = {}
    with engine.begin() as conn:
        table_names = set(inspect(conn).get_table_names())
        for table in TARGET_TABLES:
            exists = table in table_names
            before_count = _row_count_or_none(conn, table) if exists else None
            deleted = 0
            if apply and exists:
                result = conn.execute(text(f'DELETE FROM "{table}"'))
                deleted = int(result.rowcount or 0)
            summary[table] = {
                "exists": int(exists),
                "rows_before": before_count,
                "rows_deleted": deleted,
            }

        if apply and "sqlite_sequence" in table_names:
            for table in TARGET_TABLES:
                conn.execute(text("DELETE FROM sqlite_sequence WHERE name = :name"), {"name": table})

    # Recreate missing tables/schema.
    Base.metadata.create_all(bind=engine)
    return summary


def _reset_sqlite(sqlite_path: Path, apply: bool) -> Dict[str, object]:
    existed_before = sqlite_path.exists()
    if not existed_before and not apply:
        return {
            "database_mode": "SQLITE",
            "path": str(sqlite_path),
            "existed_before": False,
            "tables": {
                table: {"exists": 0, "rows_before": None, "rows_deleted": 0}
                for table in TARGET_TABLES
            },
        }

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{sqlite_path.as_posix()}", future=True)
    try:
        table_summary = _reset_tables_in_engine(engine, apply=apply)
    finally:
        engine.dispose()
    return {
        "database_mode": "SQLITE",
        "path": str(sqlite_path),
        "existed_before": existed_before,
        "tables": table_summary,
    }


def _reset_postgres(postgres_url: str, apply: bool) -> Dict[str, object]:
    engine = create_engine(postgres_url, pool_pre_ping=True, future=True)
    try:
        summary: Dict[str, Dict[str, int | None]] = {}
        with engine.begin() as conn:
            table_names = set(inspect(conn).get_table_names())
            existing_targets = [table for table in TARGET_TABLES if table in table_names]
            for table in TARGET_TABLES:
                exists = table in table_names
                before_count = _row_count_or_none(conn, table) if exists else None
                summary[table] = {
                    "exists": int(exists),
                    "rows_before": before_count,
                    "rows_deleted": 0,
                }

            if apply and existing_targets:
                joined = ", ".join(f'"{table}"' for table in existing_targets)
                conn.execute(text(f"TRUNCATE TABLE {joined} RESTART IDENTITY CASCADE"))
                for table in existing_targets:
                    before = summary[table]["rows_before"]
                    summary[table]["rows_deleted"] = int(before or 0)

        Base.metadata.create_all(bind=engine)
        return {
            "database_mode": "POSTGRES",
            "url": postgres_url,
            "tables": summary,
        }
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset TFT runtime database tables safely.")
    parser.add_argument(
        "--database-mode",
        default="auto",
        choices=["auto", "sqlite", "postgres"],
        help="Backend mode (default: auto from env).",
    )
    parser.add_argument(
        "--sqlite-path",
        default="",
        help="SQLite DB file path (optional). If omitted, env/default + state/*/tft_engine.db are scanned.",
    )
    parser.add_argument(
        "--postgres-url",
        default="",
        help="PostgreSQL SQLAlchemy URL (optional). If omitted, derived from POSTGRES_* env vars.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, dry-run only.",
    )
    args = parser.parse_args()

    mode = _detect_database_mode(args.database_mode)
    started_at = datetime.utcnow().isoformat()
    dry_run = not args.apply

    payload: Dict[str, object] = {
        "timestamp": started_at,
        "dry_run": dry_run,
        "database_mode": mode,
        "target_tables": list(TARGET_TABLES),
        "results": [],
    }

    if mode == "SQLITE":
        sqlite_targets = _discover_sqlite_targets(args.sqlite_path)
        for target in sqlite_targets:
            payload["results"].append(_reset_sqlite(target, apply=not dry_run))
    else:
        postgres_url = args.postgres_url.strip() or _postgres_url_from_env()
        payload["results"].append(_reset_postgres(postgres_url, apply=not dry_run))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
