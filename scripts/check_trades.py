#!/usr/bin/env python3.11
"""
Usage:
  python scripts/check_trades.py [--state-root state] [--logs-dir logs]
                                 [--api-url http://127.0.0.1:8000]
                                 [--window-hours 24]
                                 [--drawdown-threshold 0.15]
                                 [--loss-streak-threshold 3]
                                 [--slippage-threshold-bps 25]
                                 [--verbose]

Purpose:
  Read-only trade performance diagnostics for automation.
  This script never modifies trading state and never places trades.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


LOGGER = logging.getLogger("check_trades")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _parse_timestamp(raw: Any) -> datetime | None:
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


def _read_tail_lines(path: Path, max_lines: int, max_bytes: int = 200_000) -> list[str]:
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


def _discover_trade_dbs(state_root: Path) -> list[Path]:
    discovered: list[Path] = []
    root_db = state_root / "tft_engine.db"
    if root_db.exists():
        discovered.append(root_db)
    for candidate in sorted(state_root.glob("*/tft_engine.db")):
        if candidate.exists():
            discovered.append(candidate)
    return discovered


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _query_db_trades(db_path: Path, cutoff_utc: datetime) -> dict[str, Any]:
    payload = {
        "closed_trades": [],
        "open_positions": 0,
        "error": "",
    }
    cutoff_text = cutoff_utc.strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(str(db_path), timeout=2.0)
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn, "trades"):
            return payload

        rows = conn.execute(
            """
            SELECT
                trade_id,
                pair,
                side,
                entry_time,
                exit_time,
                pnl,
                r_multiple,
                slippage_bps,
                status
            FROM trades
            WHERE COALESCE(exit_time, entry_time) >= ?
            ORDER BY COALESCE(exit_time, entry_time) ASC
            """,
            (cutoff_text,),
        ).fetchall()
        for row in rows:
            item = dict(row)
            status = str(item.get("status") or "").lower()
            item["status"] = status
            item["timestamp"] = item.get("exit_time") or item.get("entry_time")
            if status == "closed":
                payload["closed_trades"].append(item)

        open_from_trades = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE LOWER(COALESCE(status, '')) = 'open'"
        ).fetchone()
        open_count = int(open_from_trades[0] or 0) if open_from_trades else 0

        if _table_exists(conn, "positions"):
            pos_row = conn.execute(
                "SELECT COUNT(*) FROM positions WHERE ABS(COALESCE(quantity, 0)) > 1e-12"
            ).fetchone()
            positions_count = int(pos_row[0] or 0) if pos_row else 0
            open_count = max(open_count, positions_count)

        payload["open_positions"] = open_count
        return payload
    except Exception as exc:  # pragma: no cover - defensive
        payload["error"] = str(exc)
        return payload
    finally:
        conn.close()


def _fetch_api_trades(api_url: str, timeout_s: float) -> dict[str, Any]:
    normalized = api_url.rstrip("/") + "/"
    endpoints = ("api/trades?limit=250", "trades?limit=250")
    for endpoint in endpoints:
        target = urljoin(normalized, endpoint)
        try:
            request = Request(target, headers={"Accept": "application/json"})
            with urlopen(request, timeout=timeout_s) as response:
                body = response.read().decode("utf-8", errors="ignore")
            decoded = json.loads(body)
            if isinstance(decoded, list):
                rows = decoded
            elif isinstance(decoded, dict) and isinstance(decoded.get("trades"), list):
                rows = decoded["trades"]
            else:
                rows = []
            open_positions = 0
            for row in rows:
                status = str((row or {}).get("status", "")).lower()
                if status == "open":
                    open_positions += 1
            return {
                "ok": True,
                "endpoint": target,
                "trade_count": len(rows),
                "open_positions": open_positions,
                "error": "",
            }
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            LOGGER.debug("API trade fetch failed for %s: %s", target, exc)
            continue
    return {
        "ok": False,
        "endpoint": "",
        "trade_count": 0,
        "open_positions": 0,
        "error": "API trade endpoint unreachable",
    }


def _scan_trade_logs(logs_dir: Path, max_files: int, lines_per_file: int) -> dict[str, Any]:
    patterns = ("trades*.json", "engine*.json", "engine.log", "errors*.log")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(logs_dir.glob(pattern))
    files = sorted(set(files), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    files = files[:max_files]

    event_count = 0
    for file_path in files:
        lines = _read_tail_lines(file_path, max_lines=lines_per_file)
        for line in lines:
            if not line.strip():
                continue
            matched = False
            try:
                decoded = json.loads(line)
                if isinstance(decoded, dict):
                    record = decoded.get("record", {})
                    extra = record.get("extra", {}) if isinstance(record, dict) else {}
                    event = str(extra.get("event", "")).upper()
                    if event.startswith("TRADE_"):
                        matched = True
            except Exception:
                matched = False
            if not matched:
                text = line.upper()
                if any(token in text for token in ("TRADE_OPENED", "TRADE_CLOSED", "TRADE_SUBMITTED", "TRADE_REJECTED")):
                    matched = True
            if matched:
                event_count += 1

    return {
        "files_scanned": len(files),
        "trade_events_detected": event_count,
    }


def _max_drawdown_from_pnl(pnls: list[float], starting_equity: float) -> float:
    equity = max(starting_equity, 1e-9)
    peak = equity
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = (peak - equity) / max(peak, 1e-9)
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def _loss_streak(pnls: list[float]) -> int:
    streak = 0
    max_streak = 0
    for pnl in pnls:
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only trade diagnostics")
    parser.add_argument("--state-root", default="state", help="Directory containing state DB files")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing log files")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Trading API base URL")
    parser.add_argument("--api-timeout", type=float, default=1.0, help="API timeout in seconds")
    parser.add_argument("--window-hours", type=float, default=24.0, help="Lookback window in hours")
    parser.add_argument("--drawdown-threshold", type=float, default=0.15, help="Warning drawdown threshold")
    parser.add_argument("--loss-streak-threshold", type=int, default=3, help="Consecutive loss warning threshold")
    parser.add_argument("--slippage-threshold-bps", type=float, default=25.0, help="Absolute slippage warning threshold")
    parser.add_argument("--max-log-files", type=int, default=6, help="Maximum log files to inspect")
    parser.add_argument("--log-lines-per-file", type=int, default=300, help="Tail lines read per log file")
    parser.add_argument("--starting-equity", type=float, default=10_000.0, help="Starting equity for drawdown computation")
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

    db_paths = _discover_trade_dbs(state_root)
    LOGGER.debug("Discovered %d trade DB files", len(db_paths))

    closed_trades: list[dict[str, Any]] = []
    open_positions_db = 0
    db_errors: list[str] = []

    for db_path in db_paths:
        report = _query_db_trades(db_path, cutoff)
        if report["error"]:
            db_errors.append(f"{db_path.name}: {report['error']}")
        open_positions_db += int(report["open_positions"] or 0)
        for row in report["closed_trades"]:
            row["source_db"] = str(db_path)
            closed_trades.append(row)

    closed_trades.sort(
        key=lambda row: (
            _parse_timestamp(row.get("timestamp")) or datetime.fromtimestamp(0, tz=timezone.utc),
            str(row.get("trade_id") or ""),
        )
    )

    pnl_values = [_safe_float(row.get("pnl"), 0.0) for row in closed_trades]
    trade_count = len(closed_trades)
    wins = sum(1 for pnl in pnl_values if pnl > 0)
    win_rate = (wins / trade_count) if trade_count > 0 else 0.0
    realized_pnl = sum(pnl_values)
    r_values = [
        _safe_float(row.get("r_multiple"), float("nan"))
        for row in closed_trades
        if row.get("r_multiple") is not None
    ]
    finite_r_values = [r for r in r_values if math.isfinite(r)]
    avg_r_multiple = (sum(finite_r_values) / len(finite_r_values)) if finite_r_values else 0.0
    max_drawdown = _max_drawdown_from_pnl(pnl_values, starting_equity=max(1.0, float(args.starting_equity)))
    max_loss_streak = _loss_streak(pnl_values)

    slippage_values = [
        abs(_safe_float(row.get("slippage_bps"), 0.0))
        for row in closed_trades
        if row.get("slippage_bps") is not None
    ]
    abnormal_slippage_count = sum(1 for value in slippage_values if value > float(args.slippage_threshold_bps))

    api_report = _fetch_api_trades(api_url=str(args.api_url), timeout_s=max(0.2, float(args.api_timeout)))
    logs_report = _scan_trade_logs(
        logs_dir=logs_dir,
        max_files=max(1, int(args.max_log_files)),
        lines_per_file=max(50, int(args.log_lines_per_file)),
    )

    open_positions = max(open_positions_db, int(api_report["open_positions"] or 0))

    issues: list[str] = []
    critical = False

    if db_errors:
        issues.append(f"DB read issues detected ({len(db_errors)} files)")
    if not api_report["ok"]:
        issues.append(api_report["error"])
    if trade_count == 0:
        issues.append("no trades executed in lookback window")
    if max_loss_streak >= int(args.loss_streak_threshold):
        issues.append(f"repeated losses detected (streak={max_loss_streak})")
    if max_drawdown > float(args.drawdown_threshold):
        issues.append(
            f"drawdown threshold exceeded ({max_drawdown:.4f} > {float(args.drawdown_threshold):.4f})"
        )
    if max_drawdown > float(args.drawdown_threshold) * 2.0:
        critical = True
    if abnormal_slippage_count > 0:
        issues.append(f"abnormal slippage detected ({abnormal_slippage_count} trades)")

    if trade_count == 0 and not api_report["ok"] and logs_report["trade_events_detected"] == 0:
        issues.append("no trade data available from DB/API/logs")
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
            "trade_count_last_24h": trade_count,
            "win_rate": round(win_rate, 4),
            "realized_pnl": round(realized_pnl, 6),
            "avg_r_multiple": round(avg_r_multiple, 6),
            "max_drawdown": round(max_drawdown, 6),
            "open_positions": int(open_positions),
            "max_loss_streak": int(max_loss_streak),
            "abnormal_slippage_count": int(abnormal_slippage_count),
            "api_trade_count": int(api_report["trade_count"]),
            "log_trade_events_detected": int(logs_report["trade_events_detected"]),
            "db_files_scanned": len(db_paths),
            "db_errors": db_errors,
            "api_endpoint_used": api_report["endpoint"],
            "logs_files_scanned": int(logs_report["files_scanned"]),
        },
    }

    json.dump(output, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
