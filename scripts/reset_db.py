"""Reset engine SQLite state for one or more symbol directories.

This tool is host-side (outside containers) and is intended for Docker Compose
deployments that mount `./state/<symbol>` as `/app/state`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ENGINE_TABLES: tuple[str, ...] = (
    "trades",
    "positions",
    "predictions",
    "signals",
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
    "equity_history",
    "metrics",
    "position_events",
    "statistics",
    "engine_state",
)

PAPER_TABLES: tuple[str, ...] = ("state", "positions", "trades")


@dataclass
class FileResetResult:
    path: str
    exists: bool
    archived_to: str = ""
    table_rows: Dict[str, int] = field(default_factory=dict)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_count(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
    return int(row[0]) if row else 0


def _archive_file(path: Path, archive_root: Path) -> str:
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    target = archive_root / f"{path.name}.{stamp}.bak"
    shutil.copy2(path, target)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, archive_root / f"{sidecar.name}.{stamp}.bak")
    return str(target)


def _reset_sqlite_file(
    db_path: Path,
    *,
    tables: tuple[str, ...],
    apply: bool,
    archive: bool,
    archive_root: Path,
    paper_initial_balance: float | None = None,
) -> FileResetResult:
    result = FileResetResult(path=str(db_path), exists=db_path.exists())
    if not db_path.exists():
        return result

    if archive:
        result.archived_to = _archive_file(db_path, archive_root)

    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    try:
        for table in tables:
            if not _table_exists(conn, table):
                continue
            count = _table_count(conn, table)
            result.table_rows[table] = count
            if apply:
                conn.execute(f'DELETE FROM "{table}"')

        if apply and "sqlite_sequence" in {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
            for table in tables:
                conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table,))

        if apply and paper_initial_balance is not None:
            conn.executemany(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                (
                    ("starting_balance", float(paper_initial_balance)),
                    ("balance", float(paper_initial_balance)),
                    ("realized_pnl", 0.0),
                ),
            )

        if apply:
            conn.commit()
            conn.execute("VACUUM")
    finally:
        conn.close()
    return result


def _discover_symbol_dirs(state_root: Path) -> List[Path]:
    if not state_root.exists():
        return []
    symbol_dirs: List[Path] = []
    for child in sorted(state_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "tft_engine.db").exists() or (child / "paper_trading.db").exists():
            symbol_dirs.append(child)
    return symbol_dirs


def _parse_symbols(symbols_csv: str) -> List[str]:
    raw = [token.strip().lower() for token in str(symbols_csv or "").split(",")]
    return [token for token in raw if token]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset engine DB tables (trades/positions/metrics/equity/history) for symbol state directories.",
    )
    parser.add_argument(
        "--state-root",
        default="state",
        help="Root directory that contains per-symbol folders (default: state).",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbol directory names (e.g. btc,eth,xrp). Defaults to auto-discovery.",
    )
    parser.add_argument(
        "--include-paper",
        action="store_true",
        help="Also reset paper_trading.db in each symbol directory.",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=float(os.getenv("PAPER_INITIAL_BALANCE", "10000")),
        help="Paper balance written after paper DB reset (default: PAPER_INITIAL_BALANCE or 10000).",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create a timestamped .bak copy before modifying any DB file.",
    )
    parser.add_argument(
        "--archive-dir",
        default="state/archive",
        help="Archive destination directory when --archive is enabled.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag the command runs in dry-run mode.",
    )
    args = parser.parse_args()

    state_root = Path(args.state_root).resolve()
    archive_root = Path(args.archive_dir).resolve()
    requested = set(_parse_symbols(args.symbols))
    symbol_dirs = _discover_symbol_dirs(state_root)
    if requested:
        symbol_dirs = [path for path in symbol_dirs if path.name.lower() in requested]

    summary: Dict[str, object] = {
        "timestamp": datetime.utcnow().isoformat(),
        "dry_run": not args.apply,
        "state_root": str(state_root),
        "symbol_dirs": [path.name for path in symbol_dirs],
        "engine_resets": [],
        "paper_resets": [],
    }

    for symbol_dir in symbol_dirs:
        engine_db = symbol_dir / "tft_engine.db"
        engine_result = _reset_sqlite_file(
            engine_db,
            tables=ENGINE_TABLES,
            apply=bool(args.apply),
            archive=bool(args.archive),
            archive_root=archive_root / symbol_dir.name / "engine",
        )
        summary["engine_resets"].append(
            {
                "symbol_dir": symbol_dir.name,
                "result": engine_result.__dict__,
            }
        )

        if args.include_paper:
            paper_db = symbol_dir / "paper_trading.db"
            paper_result = _reset_sqlite_file(
                paper_db,
                tables=PAPER_TABLES,
                apply=bool(args.apply),
                archive=bool(args.archive),
                archive_root=archive_root / symbol_dir.name / "paper",
                paper_initial_balance=float(args.initial_balance),
            )
            summary["paper_resets"].append(
                {
                    "symbol_dir": symbol_dir.name,
                    "result": paper_result.__dict__,
                }
            )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

