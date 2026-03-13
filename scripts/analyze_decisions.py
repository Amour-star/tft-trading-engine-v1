#!/usr/bin/env python3.11
"""
Usage:
  python scripts/analyze_decisions.py [--state-root state] [--logs-dir logs]
                                      [--window-hours 24]
                                      [--signal-threshold 0.45]
                                      [--rejection-threshold 0.90]
                                      [--max-db-events 5000]
                                      [--verbose]

Purpose:
  Read-only decision-engine diagnostics for automation.
  This script never modifies trading state and never places trades.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("analyze_decisions")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _decode_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _unwrap_engine_state(raw_value: Any) -> Any:
    parsed = _decode_jsonish(raw_value)
    if isinstance(parsed, dict) and "value" in parsed:
        return parsed.get("value")
    return parsed


def _discover_state_dbs(state_root: Path) -> list[Path]:
    dbs: list[Path] = []
    root_db = state_root / "tft_engine.db"
    if root_db.exists():
        dbs.append(root_db)
    for child in sorted(state_root.glob("*/tft_engine.db")):
        if child.exists():
            dbs.append(child)
    return dbs


def _read_tail_lines(path: Path, max_lines: int, max_bytes: int = 250_000) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        size = path.stat().st_size
        read_size = min(size, max_bytes)
        with path.open("rb") as handle:
            if read_size < size:
                handle.seek(-read_size, 2)
            blob = handle.read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    lines = blob.splitlines()
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _query_decision_data(
    db_path: Path,
    cutoff_utc: datetime,
    max_events: int,
) -> dict[str, Any]:
    result = {
        "decision_rows": [],
        "signal_rows": [],
        "engine_state": {},
        "error": "",
    }
    cutoff_text = cutoff_utc.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(str(db_path), timeout=2.0)
    conn.row_factory = sqlite3.Row
    try:
        if _table_exists(conn, "decision_events"):
            decision_rows = conn.execute(
                """
                SELECT timestamp, status, reason, regime, top_candidates_json
                FROM decision_events
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cutoff_text, max_events),
            ).fetchall()
            result["decision_rows"] = [dict(row) for row in decision_rows]

        if _table_exists(conn, "signals"):
            signal_rows = conn.execute(
                """
                SELECT timestamp, signal_score, payload
                FROM signals
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cutoff_text, max_events),
            ).fetchall()
            result["signal_rows"] = [dict(row) for row in signal_rows]

        if _table_exists(conn, "engine_state"):
            keys = (
                "tft_disabled",
                "active_model_name",
                "trade_rejection_reasons",
                "last_cycle_reason",
            )
            placeholder = ",".join("?" for _ in keys)
            rows = conn.execute(
                f"SELECT key, value FROM engine_state WHERE key IN ({placeholder})",
                keys,
            ).fetchall()
            result["engine_state"] = {str(row["key"]): _unwrap_engine_state(row["value"]) for row in rows}

        return result
    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = str(exc)
        return result
    finally:
        conn.close()


def _scan_decision_logs(logs_dir: Path, max_files: int, lines_per_file: int) -> dict[str, int]:
    files = sorted(
        set(list(logs_dir.glob("engine*.json")) + list(logs_dir.glob("engine.log"))),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )[:max_files]

    fallback_mentions = 0
    regime_none_mentions = 0
    for file_path in files:
        for line in _read_tail_lines(file_path, max_lines=lines_per_file):
            upper = line.upper()
            if "XGB_META_FALLBACK" in upper or "TFT_FORCE_DISABLE" in upper or "TFT DISABLED" in upper:
                fallback_mentions += 1
            if "REGIME" in upper and "NONE" in upper:
                regime_none_mentions += 1
    return {
        "files_scanned": len(files),
        "fallback_mentions": fallback_mentions,
        "regime_none_mentions": regime_none_mentions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only decision engine diagnostics")
    parser.add_argument("--state-root", default="state", help="Directory containing state DB files")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing log files")
    parser.add_argument("--window-hours", type=float, default=24.0, help="Lookback window in hours")
    parser.add_argument("--signal-threshold", type=float, default=0.45, help="Low-signal threshold")
    parser.add_argument("--rejection-threshold", type=float, default=0.90, help="Warning rejection-rate threshold")
    parser.add_argument("--max-db-events", type=int, default=5000, help="Max rows per table per DB")
    parser.add_argument("--max-log-files", type=int, default=6, help="Max log files to inspect")
    parser.add_argument("--log-lines-per-file", type=int, default=350, help="Tail lines per file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose stderr logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(0.0, float(args.window_hours)))

    state_root = Path(args.state_root).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    db_paths = _discover_state_dbs(state_root)

    reason_counts: Counter[str] = Counter()
    decision_cycles = 0
    trades_opened = 0
    regime_none_count = 0
    safety_blocked = 0
    fallback_model_active = False
    low_signal_score_count = 0
    signal_rows_scanned = 0
    db_errors: list[str] = []
    rejection_reason_counts: Counter[str] = Counter()

    for db_path in db_paths:
        report = _query_decision_data(
            db_path=db_path,
            cutoff_utc=cutoff,
            max_events=max(1, int(args.max_db_events)),
        )
        if report["error"]:
            db_errors.append(f"{db_path.name}: {report['error']}")
            continue

        for row in report["decision_rows"]:
            decision_cycles += 1
            status = str(row.get("status") or "").strip().lower()
            reason = str(row.get("reason") or "").strip()
            if status == "trade_opened":
                trades_opened += 1
            if reason:
                reason_counts[reason] += 1
            if reason.startswith("safety_can_trade_blocked"):
                safety_blocked += 1
            regime = row.get("regime")
            if regime is None or str(regime).strip() == "":
                regime_none_count += 1

        for row in report["signal_rows"]:
            signal_rows_scanned += 1
            signal_score = _safe_float(row.get("signal_score"), 0.0)
            if signal_score < float(args.signal_threshold):
                low_signal_score_count += 1

            payload = _decode_jsonish(row.get("payload"))
            if isinstance(payload, dict):
                reason = str(payload.get("reason") or "").lower()
                if "fallback" in reason:
                    fallback_model_active = True
                block_reasons = payload.get("block_reasons")
                if isinstance(block_reasons, list):
                    for block_reason in block_reasons:
                        name = str(block_reason or "").strip()
                        if name:
                            rejection_reason_counts[name] += 1

        engine_state = report["engine_state"]
        tft_disabled = bool(engine_state.get("tft_disabled", False))
        active_model_name = str(engine_state.get("active_model_name") or "").strip().lower()
        if tft_disabled or active_model_name in {"xgb_meta_fallback", "tft_disabled"}:
            fallback_model_active = True

        trade_rejection_reasons = engine_state.get("trade_rejection_reasons")
        if isinstance(trade_rejection_reasons, dict):
            for key, value in trade_rejection_reasons.items():
                rejection_reason_counts[str(key)] += int(_safe_float(value, 0.0))

    blocked_cycles = max(0, decision_cycles - trades_opened)
    rejection_rate = (blocked_cycles / decision_cycles) if decision_cycles > 0 else 0.0
    low_signal_ratio = (low_signal_score_count / signal_rows_scanned) if signal_rows_scanned > 0 else 0.0

    log_report = _scan_decision_logs(
        logs_dir=logs_dir,
        max_files=max(1, int(args.max_log_files)),
        lines_per_file=max(50, int(args.log_lines_per_file)),
    )
    fallback_mentions = int(log_report["fallback_mentions"])
    if fallback_mentions > 0:
        fallback_model_active = True

    issues: list[str] = []
    critical = False

    if db_errors:
        issues.append(f"decision DB read issues detected ({len(db_errors)} files)")

    if decision_cycles == 0:
        issues.append("no decision cycles found in lookback window")

    if safety_blocked > 0:
        issues.append("safety gates blocking trades")

    if regime_none_count > 0 and decision_cycles > 0:
        ratio = regime_none_count / decision_cycles
        if ratio >= 0.2:
            issues.append("regime detection returning None frequently")

    if fallback_model_active:
        issues.append("fallback model usage detected")

    if rejection_rate > float(args.rejection_threshold):
        issues.append(f"high rejection rate ({rejection_rate:.3f})")

    if signal_rows_scanned > 0 and low_signal_ratio > 0.7:
        issues.append("signal_score below threshold for most recent signals")

    if rejection_rate >= 0.98 and decision_cycles >= 50:
        critical = True
    if safety_blocked >= 100:
        critical = True

    if critical:
        status = "critical"
        exit_code = 2
    elif issues:
        status = "warning"
        exit_code = 1
    else:
        status = "ok"
        exit_code = 0

    output = {
        "status": status,
        "issues": issues,
        "metrics": {
            "decision_cycles": int(decision_cycles),
            "trades_opened": int(trades_opened),
            "blocked_cycles": int(blocked_cycles),
            "block_reason_counts": dict(reason_counts),
            "rejection_rate": round(rejection_rate, 6),
            "safety_can_trade_blocked": int(safety_blocked),
            "regime_none_count": int(regime_none_count),
            "fallback_model_active": bool(fallback_model_active),
            "fallback_mentions": int(fallback_mentions),
            "low_signal_score_count": int(low_signal_score_count),
            "low_signal_score_ratio": round(low_signal_ratio, 6),
            "signal_rows_scanned": int(signal_rows_scanned),
            "trade_rejection_reason_counts": dict(rejection_reason_counts),
            "db_files_scanned": len(db_paths),
            "db_errors": db_errors,
            "log_files_scanned": int(log_report["files_scanned"]),
            "log_regime_none_mentions": int(log_report["regime_none_mentions"]),
        },
    }

    json.dump(output, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
