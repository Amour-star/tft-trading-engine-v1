#!/usr/bin/env python3.11
"""
Usage:
  python scripts/daily_model_review.py [--saved-models-dir saved_models]
                                       [--state-root state]
                                       [--logs-dir logs]
                                       [--stale-days 7]
                                       [--verbose]

Purpose:
  Read-only model/training health diagnostics for automation.
  This script never modifies model state or trading state.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("daily_model_review")


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return float(parsed)


def _parse_datetime(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    for parser in (
        lambda v: datetime.fromisoformat(v),
        lambda v: datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f"),
        lambda v: datetime.strptime(v, "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            parsed = parser(normalized)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _read_tail_lines(path: Path, max_lines: int, max_bytes: int = 300_000) -> list[str]:
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


def _discover_state_dbs(state_root: Path) -> list[Path]:
    dbs: list[Path] = []
    root_db = state_root / "tft_engine.db"
    if root_db.exists():
        dbs.append(root_db)
    for child in sorted(state_root.glob("*/tft_engine.db")):
        if child.exists():
            dbs.append(child)
    return dbs


def _parse_checkpoint_validation_loss(model_dir: Path) -> float | None:
    pattern = re.compile(r"val_loss=([0-9]*\.?[0-9]+)")
    candidates = sorted(model_dir.glob("best-*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for ckpt in candidates:
        match = pattern.search(ckpt.name)
        if not match:
            continue
        parsed = _safe_float(match.group(1), None)
        if parsed is not None:
            return parsed
    return None


def _model_dir_summary(model_dir: Path) -> dict[str, Any]:
    metadata = {}
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        try:
            parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                metadata = parsed
        except Exception:
            metadata = {}

    info_path = model_dir / "info.txt"
    info_payload = {}
    if info_path.exists():
        try:
            for line in info_path.read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                info_payload[key.strip()] = value.strip()
        except Exception:
            info_payload = {}

    validation_loss = (
        _safe_float(metadata.get("validation_loss"), None)
        if isinstance(metadata, dict)
        else None
    )
    if validation_loss is None:
        validation_loss = _parse_checkpoint_validation_loss(model_dir)

    training_date = _parse_datetime((metadata or {}).get("trained_at"))
    if training_date is None and info_payload.get("trained_at"):
        training_date = _parse_datetime(info_payload.get("trained_at"))
    if training_date is None:
        latest_mtime = max(
            (child.stat().st_mtime for child in model_dir.glob("*") if child.exists()),
            default=model_dir.stat().st_mtime,
        )
        training_date = datetime.fromtimestamp(latest_mtime, tz=timezone.utc)

    size_bytes = 0
    for child in model_dir.rglob("*"):
        if child.is_file():
            try:
                size_bytes += child.stat().st_size
            except OSError:
                continue

    return {
        "version": model_dir.name,
        "training_date": training_date,
        "validation_loss": validation_loss,
        "model_size_bytes": size_bytes,
        "metadata": metadata,
    }


def _query_registry_data(db_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "db_path": str(db_path),
        "model_metrics_rows": 0,
        "model_versions_rows": 0,
        "active_model_metric": None,
        "active_model_version": None,
        "fallback_active": False,
        "error": "",
    }
    conn = sqlite3.connect(str(db_path), timeout=2.0)
    conn.row_factory = sqlite3.Row
    try:
        if _table_exists(conn, "model_metrics"):
            row_count = conn.execute("SELECT COUNT(*) FROM model_metrics").fetchone()
            payload["model_metrics_rows"] = int(row_count[0] or 0) if row_count else 0
            active = conn.execute(
                """
                SELECT model_version, trained_at, validation_loss, is_active
                FROM model_metrics
                WHERE is_active = 1
                ORDER BY trained_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
            if active:
                payload["active_model_metric"] = dict(active)

        if _table_exists(conn, "model_versions"):
            row_count = conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()
            payload["model_versions_rows"] = int(row_count[0] or 0) if row_count else 0
            active = conn.execute(
                """
                SELECT model_type, version, path, is_active
                FROM model_versions
                WHERE is_active = 1
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
            if active:
                payload["active_model_version"] = dict(active)

        if _table_exists(conn, "engine_state"):
            rows = conn.execute(
                """
                SELECT key, value
                FROM engine_state
                WHERE key IN ('tft_disabled', 'active_model_name')
                """
            ).fetchall()
            state = {str(row["key"]): row["value"] for row in rows}
            tft_disabled_raw = state.get("tft_disabled")
            active_model_raw = state.get("active_model_name")

            tft_disabled = False
            if tft_disabled_raw is not None:
                try:
                    parsed = json.loads(str(tft_disabled_raw))
                    if isinstance(parsed, dict) and "value" in parsed:
                        tft_disabled = bool(parsed["value"])
                    else:
                        tft_disabled = bool(parsed)
                except Exception:
                    tft_disabled = False

            active_model_name = ""
            if active_model_raw is not None:
                try:
                    parsed = json.loads(str(active_model_raw))
                    if isinstance(parsed, dict) and "value" in parsed:
                        active_model_name = str(parsed["value"] or "")
                    else:
                        active_model_name = str(parsed or "")
                except Exception:
                    active_model_name = str(active_model_raw or "")

            payload["fallback_active"] = bool(
                tft_disabled or active_model_name.strip().lower() in {"xgb_meta_fallback", "tft_disabled"}
            )
        return payload
    except Exception as exc:  # pragma: no cover - defensive
        payload["error"] = str(exc)
        return payload
    finally:
        conn.close()


def _scan_training_logs(logs_dir: Path, max_files: int, lines_per_file: int) -> dict[str, Any]:
    files = sorted(
        set(list(logs_dir.glob("engine*.json")) + list(logs_dir.glob("engine.log")) + list(logs_dir.glob("errors*.log"))),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )[:max_files]

    tokens = ("TFT MODEL TRAINING", "TRAINING TFT MODEL", "train_model:main")
    latest_training_log_ts: datetime | None = None

    for file_path in files:
        for line in _read_tail_lines(file_path, max_lines=lines_per_file):
            upper = line.upper()
            if not any(token in upper for token in tokens):
                continue

            candidate_ts: datetime | None = None
            try:
                decoded = json.loads(line)
                if isinstance(decoded, dict):
                    record = decoded.get("record", {})
                    time_info = record.get("time", {}) if isinstance(record, dict) else {}
                    candidate_ts = _parse_datetime(time_info.get("repr"))
            except Exception:
                candidate_ts = None

            if candidate_ts is None:
                prefix = line[:32]
                candidate_ts = _parse_datetime(prefix)

            if candidate_ts is None:
                continue
            if latest_training_log_ts is None or candidate_ts > latest_training_log_ts:
                latest_training_log_ts = candidate_ts

    return {
        "files_scanned": len(files),
        "latest_training_log_ts": latest_training_log_ts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only daily model review")
    parser.add_argument("--saved-models-dir", default="saved_models", help="Directory containing model artifacts")
    parser.add_argument("--state-root", default="state", help="Directory containing state DBs")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing logs")
    parser.add_argument("--stale-days", type=float, default=7.0, help="Model staleness threshold")
    parser.add_argument("--max-log-files", type=int, default=6, help="Max log files to inspect")
    parser.add_argument("--log-lines-per-file", type=int, default=350, help="Tail lines per log file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose stderr logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    now = datetime.now(timezone.utc)
    saved_models_dir = Path(args.saved_models_dir).resolve()
    state_root = Path(args.state_root).resolve()
    logs_dir = Path(args.logs_dir).resolve()

    model_dirs = [path for path in sorted(saved_models_dir.glob("*")) if path.is_dir()]
    model_summaries = [_model_dir_summary(path) for path in model_dirs]
    model_summaries.sort(
        key=lambda item: (item.get("training_date") or datetime.fromtimestamp(0, tz=timezone.utc)),
        reverse=True,
    )
    latest_model = model_summaries[0] if model_summaries else None

    state_dbs = _discover_state_dbs(state_root)
    registry_rows: list[dict[str, Any]] = []
    registry_errors: list[str] = []
    fallback_model_active = False
    for db_path in state_dbs:
        row = _query_registry_data(db_path)
        registry_rows.append(row)
        if row.get("error"):
            registry_errors.append(f"{db_path.name}: {row['error']}")
        if row.get("fallback_active"):
            fallback_model_active = True

    active_registry_model = None
    active_metric_model = None
    total_model_metrics_rows = 0
    total_model_versions_rows = 0
    for row in registry_rows:
        total_model_metrics_rows += int(row.get("model_metrics_rows") or 0)
        total_model_versions_rows += int(row.get("model_versions_rows") or 0)
        if active_registry_model is None and isinstance(row.get("active_model_version"), dict):
            active_registry_model = row["active_model_version"]
        if active_metric_model is None and isinstance(row.get("active_model_metric"), dict):
            active_metric_model = row["active_model_metric"]

    log_report = _scan_training_logs(
        logs_dir=logs_dir,
        max_files=max(1, int(args.max_log_files)),
        lines_per_file=max(50, int(args.log_lines_per_file)),
    )

    latest_model_version = latest_model["version"] if latest_model else None
    training_date = latest_model["training_date"] if latest_model else None
    validation_loss = latest_model["validation_loss"] if latest_model else None
    model_size_bytes = int(latest_model["model_size_bytes"]) if latest_model else 0

    if validation_loss is None and isinstance(active_metric_model, dict):
        validation_loss = _safe_float(active_metric_model.get("validation_loss"), None)

    model_age_days = None
    if training_date is not None:
        model_age_days = (now - training_date).total_seconds() / 86400.0

    issues: list[str] = []
    critical = False

    if latest_model is None:
        issues.append("no saved model artifacts found")
        critical = True
    if model_age_days is not None and model_age_days > float(args.stale_days):
        issues.append(
            f"latest model is older than threshold ({model_age_days:.2f}d > {float(args.stale_days):.2f}d)"
        )
    if fallback_model_active:
        issues.append("fallback model active in engine state")
    if validation_loss is None:
        issues.append("missing validation metrics for latest model")
    if total_model_metrics_rows == 0 and total_model_versions_rows == 0:
        issues.append("model registry tables are empty")
    if registry_errors:
        issues.append(f"registry read issues detected ({len(registry_errors)} files)")

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
            "latest_model_version": latest_model_version,
            "training_date": training_date.isoformat() if isinstance(training_date, datetime) else None,
            "model_age_days": round(model_age_days, 4) if model_age_days is not None else None,
            "validation_loss": validation_loss,
            "model_size": model_size_bytes,
            "model_size_mb": round(model_size_bytes / (1024 * 1024), 4) if model_size_bytes > 0 else 0.0,
            "fallback_model_active": bool(fallback_model_active),
            "active_registry_model": active_registry_model,
            "active_model_metric": active_metric_model,
            "saved_models_count": len(model_summaries),
            "registry_model_metrics_rows": int(total_model_metrics_rows),
            "registry_model_versions_rows": int(total_model_versions_rows),
            "registry_errors": registry_errors,
            "db_files_scanned": len(state_dbs),
            "training_log_last_seen": (
                log_report["latest_training_log_ts"].isoformat()
                if isinstance(log_report.get("latest_training_log_ts"), datetime)
                else None
            ),
            "training_log_files_scanned": int(log_report["files_scanned"]),
        },
    }

    json.dump(output, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
