from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from docker import DockerClient
from docker.errors import DockerException, NotFound


STATE_ROOT = Path("/app/state")
MONITOR_LOG_PATH = STATE_ROOT / "monitor.log"
CHECK_INTERVAL_MINUTES = 30
NO_TRADE_THRESHOLD_HOURS = 6
REPEATED_FAILURE_THRESHOLD = 2
MAX_TFT_RETRIES = 1
NON_RESTARTABLE_ISSUES = {"No trades for > 6 hours"}

ENGINE_NAMES: Dict[str, List[str]] = {
    "btc": ["tft-engine-btc", "tft-trading-engine-engine-btc-1"],
    "eth": ["tft-engine-eth", "tft-trading-engine-engine-eth-1"],
    "xrp": ["tft-engine-xrp", "tft-trading-engine-engine-xrp-1"],
    "doge": ["tft-engine-doge", "tft-trading-engine-engine-doge-1"],
}

ERROR_PATTERNS = [
    "unable to open database file",
    "database disk image is malformed",
    "Out of memory",
    "EMERGENCY_CLEANUP_FAILED",
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
                    restartable_issues = [
                        issue for issue in result.issues if issue not in NON_RESTARTABLE_ISSUES
                    ]
                    event = "issues_detected" if restartable_issues else "warning_detected"
                    self._write_report(
                        event,
                        f"{symbol} ({result.container_name}): {result.issues}",
                    )
                    if restartable_issues:
                        self.failure_counts[symbol] += 1
                        self._restart_container(result.container_name)
                        if self.failure_counts[symbol] >= REPEATED_FAILURE_THRESHOLD:
                            self._disable_tft_for_symbol(symbol, restartable_issues)
                    else:
                        self.failure_counts[symbol] = 0
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
        if hasattr(container, "reload"):
            try:
                container.reload()
            except DockerException:
                pass
        status = (container.status or "").lower()
        if status != "running":
            issues.append("Engine not running")

        health_status = self._container_health_status(container)
        since_dt = datetime.now(timezone.utc) - timedelta(minutes=CHECK_INTERVAL_MINUTES)
        log_text = self._get_container_logs(container.name, since_dt)
        logs_checked = len(log_text.splitlines()) if log_text else 0
        if health_status != "healthy":
            log_text_lower = log_text.lower()
            for pattern in ERROR_PATTERNS:
                if pattern.lower() in log_text_lower:
                    issues.append(f"log_match:{pattern}")

        no_trade_issue = self._no_trade_issue(symbol, container)
        if no_trade_issue:
            issues.append(no_trade_issue)

        return CheckResult(
            symbol=symbol,
            container_name=container.name,
            issues=issues,
            logs_checked=logs_checked,
        )

    @staticmethod
    def _container_health_status(container) -> str:
        try:
            state = getattr(container, "attrs", {}).get("State", {})
            health = state.get("Health") or {}
            status = str(health.get("Status", "")).strip().lower()
            return status or "none"
        except Exception:
            return "none"

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

    def _no_trade_issue(self, symbol: str, container=None) -> Optional[str]:
        now_utc = datetime.now(timezone.utc)
        db_candidates = [
            STATE_ROOT / symbol / "trades.db",
            STATE_ROOT / symbol / "tft_engine.db",
        ]
        db_path = next((path for path in db_candidates if path.exists()), None)
        if db_path is None:
            if self._is_recent_symbol_start(symbol, container, now_utc):
                return None
            return "No trades for > 6 hours"

        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS trade_count,
                    MAX(COALESCE(exit_time, entry_time)) AS last_trade_ts
                FROM trades
                """
            )
            row = cursor.fetchone()
        except Exception:
            if self._is_recent_symbol_start(symbol, container, now_utc):
                return None
            return "No trades for > 6 hours"
        finally:
            try:
                conn.close()
            except Exception:
                pass

        trade_count = 0 if row is None else int(row["trade_count"] or 0)
        if trade_count <= 0:
            if self._is_recent_symbol_start(symbol, container, now_utc):
                return None
            return "No trades for > 6 hours"

        last_trade_ts = None if row is None else row["last_trade_ts"]
        if not last_trade_ts:
            if self._is_recent_symbol_start(symbol, container, now_utc):
                return None
            return "No trades for > 6 hours"

        parsed_ts = self._parse_timestamp(str(last_trade_ts))
        if parsed_ts is None:
            if self._is_recent_symbol_start(symbol, container, now_utc):
                return None
            return "No trades for > 6 hours"

        age = now_utc - parsed_ts
        if age > timedelta(hours=NO_TRADE_THRESHOLD_HOURS):
            return "No trades for > 6 hours"
        return None

    def _is_recent_symbol_start(self, symbol: str, container, now_utc: datetime) -> bool:
        latest_reference = self._latest_symbol_reference(symbol, container)
        if latest_reference is None:
            return False
        return (now_utc - latest_reference) <= timedelta(hours=NO_TRADE_THRESHOLD_HOURS)

    def _latest_symbol_reference(self, symbol: str, container) -> Optional[datetime]:
        symbol_state = STATE_ROOT / symbol
        candidates = [symbol_state, symbol_state / "tft_engine.db", symbol_state / "trades.db"]
        references: List[datetime] = []
        started_at = self._container_started_at(container)
        if started_at is not None:
            references.append(started_at)
        for path in candidates:
            if not path.exists():
                continue
            try:
                references.append(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
            except OSError:
                continue
        if not references:
            return None
        return max(references)

    def _container_started_at(self, container) -> Optional[datetime]:
        if container is None:
            return None
        try:
            state = getattr(container, "attrs", {}).get("State", {})
            raw = str(state.get("StartedAt", "")).strip()
            if not raw:
                return None
            if raw.endswith("Z"):
                raw = raw[:-1]
            if "." in raw:
                raw = raw.split(".", 1)[0]
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

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
        """Log repeated failures but do NOT write filesystem disable flags.
        The engine handles degraded mode internally. Writing flags caused a
        death-spiral where the monitor's 'fix' prevented recovery."""
        self._write_report(
            "repeated_failure_logged",
            f"{symbol}: {issues} (failure_count={self.failure_counts.get(symbol, 0)}). "
            f"NOT writing tft_disabled.flag — engine manages its own fallback mode.",
        )
        # Still restart the container to give it a fresh chance
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
