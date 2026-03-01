from __future__ import annotations

import sqlite3
import time
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from apscheduler.schedulers.background import BackgroundScheduler
from docker import DockerClient
from docker.errors import DockerException, NotFound


STATE_ROOT = Path("/app/state")
MONITOR_LOG_PATH = STATE_ROOT / "monitor.log"
CHECK_INTERVAL_MINUTES = 30
NO_TRADE_THRESHOLD_HOURS = 6
REPEATED_FAILURE_THRESHOLD = 2
MAX_TFT_RETRIES = 1

ENGINE_NAMES: Dict[str, List[str]] = {
    "btc": ["tft-engine-btc", "tft-trading-engine-engine-btc-1"],
    "eth": ["tft-engine-eth", "tft-trading-engine-engine-eth-1"],
    "xrp": ["tft-engine-xrp", "tft-trading-engine-engine-xrp-1"],
    "doge": ["tft-engine-doge", "tft-trading-engine-engine-doge-1"],
}

ERROR_PATTERNS = [
    "ERROR",
    "Model load failed",
    "CUDA error",
    "Out of memory",
]


@dataclass
class CheckResult:
    symbol: str
    container_name: str
    issues: List[str]
    logs_checked: int


class EngineMonitor:
    def __init__(self) -> None:
        self.client: DockerClient = DockerClient.from_env()
        self.failure_counts: Dict[str, int] = {symbol: 0 for symbol in ENGINE_NAMES}
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        self._write_report("monitor_started", "Engine monitor initialized")

    def run_once(self) -> None:
        for symbol in ENGINE_NAMES:
            try:
                result = self._check_symbol(symbol)
                if result.issues:
                    self.failure_counts[symbol] += 1
                    self._write_report(
                        "issues_detected",
                        f"{symbol} ({result.container_name}): {result.issues}",
                    )
                    self._restart_container(result.container_name)
                    if self.failure_counts[symbol] >= REPEATED_FAILURE_THRESHOLD:
                        self._disable_tft_for_symbol(symbol, result.issues)
                else:
                    self.failure_counts[symbol] = 0
                    self._write_report(
                        "healthy",
                        f"{symbol} ({result.container_name}) healthy in last {CHECK_INTERVAL_MINUTES}m",
                    )
            except Exception as exc:
                self.failure_counts[symbol] += 1
                self._write_report("monitor_error", f"{symbol}: {exc}")

    def _check_symbol(self, symbol: str) -> CheckResult:
        container = self._resolve_container(symbol)
        if container is None:
            return CheckResult(
                symbol=symbol,
                container_name=ENGINE_NAMES[symbol][0],
                issues=["Engine not running"],
                logs_checked=0,
            )

        issues: List[str] = []
        status = (container.status or "").lower()
        if status != "running":
            issues.append("Engine not running")

        since_dt = datetime.now(timezone.utc) - timedelta(minutes=CHECK_INTERVAL_MINUTES)
        log_text = self._get_container_logs(container.name, since_dt)
        logs_checked = len(log_text.splitlines()) if log_text else 0
        for pattern in ERROR_PATTERNS:
            if pattern in log_text:
                issues.append(f"log_match:{pattern}")

        no_trade_issue = self._no_trade_issue(symbol)
        if no_trade_issue:
            issues.append(no_trade_issue)

        return CheckResult(
            symbol=symbol,
            container_name=container.name,
            issues=issues,
            logs_checked=logs_checked,
        )

    def _resolve_container(self, symbol: str):
        for name in ENGINE_NAMES[symbol]:
            try:
                return self.client.containers.get(name)
            except NotFound:
                continue
        return None

    def _get_container_logs(self, container_name: str, since_dt: datetime) -> str:
        try:
            container = self.client.containers.get(container_name)
            raw = container.logs(since=since_dt, stdout=True, stderr=True)
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="ignore")
            return str(raw)
        except DockerException as exc:
            self._write_report("log_read_error", f"{container_name}: {exc}")
            return ""

    def _no_trade_issue(self, symbol: str) -> Optional[str]:
        self._ensure_trades_db_alias(symbol)
        db_candidates = [
            STATE_ROOT / symbol / "trades.db",
            STATE_ROOT / symbol / "tft_engine.db",
        ]
        db_path = next((path for path in db_candidates if path.exists()), None)
        if db_path is None:
            return "No trades for > 6 hours"

        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT MAX(COALESCE(exit_time, entry_time)) AS last_trade_ts
                FROM trades
                """
            )
            row = cursor.fetchone()
        except Exception:
            return "No trades for > 6 hours"
        finally:
            try:
                conn.close()
            except Exception:
                pass

        last_trade_ts = None if row is None else row["last_trade_ts"]
        if not last_trade_ts:
            return "No trades for > 6 hours"

        parsed_ts = self._parse_timestamp(str(last_trade_ts))
        if parsed_ts is None:
            return "No trades for > 6 hours"

        age = datetime.now(timezone.utc) - parsed_ts
        if age > timedelta(hours=NO_TRADE_THRESHOLD_HOURS):
            return "No trades for > 6 hours"
        return None

    def _ensure_trades_db_alias(self, symbol: str) -> None:
        symbol_state = STATE_ROOT / symbol
        source = symbol_state / "tft_engine.db"
        target = symbol_state / "trades.db"
        if not source.exists():
            return

        if target.exists():
            try:
                _ = target.stat()
                return
            except Exception:
                try:
                    target.unlink(missing_ok=True)
                except Exception:
                    return

        try:
            shutil.copy2(source, target)
            self._write_report("trades_db_copied", f"{symbol}: {target} from {source}")
        except Exception as exc:
            self._write_report("trades_db_alias_failed", f"{symbol}: {exc}")

    @staticmethod
    def _parse_timestamp(raw: str) -> Optional[datetime]:
        text = str(raw).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        for parser in (datetime.fromisoformat,):
            try:
                parsed = parser(normalized)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except Exception:
                continue
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _restart_container(self, container_name: str) -> None:
        try:
            container = self.client.containers.get(container_name)
            container.restart(timeout=30)
            self._write_report("container_restarted", container_name)
        except DockerException as exc:
            self._write_report("restart_failed", f"{container_name}: {exc}")

    def _disable_tft_for_symbol(self, symbol: str, issues: List[str]) -> None:
        symbol_state = STATE_ROOT / symbol
        symbol_state.mkdir(parents=True, exist_ok=True)
        flag_path = symbol_state / "tft_disabled.flag"
        content = (
            f"disabled_at={datetime.utcnow().isoformat()}Z\n"
            f"reason=repeated_monitor_failure\n"
            f"issues={','.join(issues)}\n"
            f"max_tft_retries={MAX_TFT_RETRIES}\n"
            "tft_auto_load_disabled=true\n"
        )
        flag_path.write_text(content, encoding="utf-8")
        self._write_report("tft_disabled", f"{symbol}: {flag_path}")
        container = self._resolve_container(symbol)
        if container is not None:
            self._restart_container(container.name)

    @staticmethod
    def _write_report(event: str, message: str) -> None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        line = f"{timestamp} | {event} | {message}\n"
        with MONITOR_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)


def main() -> None:
    monitor = EngineMonitor()
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        monitor.run_once,
        "interval",
        minutes=CHECK_INTERVAL_MINUTES,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )
    scheduler.start()
    monitor.run_once()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
