from __future__ import annotations

import time
from threading import Event, Thread
from typing import Any, Callable, Dict, Iterable, Optional

from loguru import logger

from .kucoin_rest import KuCoinRestClient

StatusProvider = Callable[[], Dict[str, Dict[str, Any]]]
HaltCallback = Callable[[str, Dict[str, Any]], None]
RecoverCallback = Callable[[Dict[str, Any]], None]


class HeartbeatMonitor:
    """Continuously checks market-data liveness and triggers trading halt."""

    def __init__(
        self,
        symbols: Iterable[str],
        rest_client: KuCoinRestClient,
        ws_active_provider: Callable[[], bool],
        latest_ticker_provider: StatusProvider,
        on_halt: HaltCallback,
        on_recover: Optional[RecoverCallback] = None,
        interval_seconds: float = 2.0,
        max_age_seconds: float = 30.0,
    ) -> None:
        self._symbols = sorted({str(sym).upper() for sym in symbols if str(sym).strip()})
        self._rest = rest_client
        self._ws_active_provider = ws_active_provider
        self._latest_ticker_provider = latest_ticker_provider
        self._on_halt = on_halt
        self._on_recover = on_recover
        self._interval_seconds = max(0.5, float(interval_seconds))
        self._max_age_seconds = max(5.0, float(max_age_seconds))
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._halt_active = False
        self._latest_heartbeat: Dict[str, Any] = {}
        self._healthy: bool = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="market-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def latest_heartbeat(self) -> Dict[str, Any]:
        return dict(self._latest_heartbeat)

    @property
    def is_healthy(self) -> bool:
        return bool(self._healthy)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            api_ok = self._rest.is_reachable()
            ws_ok = bool(self._ws_active_provider())
            ticker_map = self._latest_ticker_provider() or {}

            stale_symbols = []
            missing_symbols = []
            for symbol in self._symbols:
                item = ticker_map.get(symbol)
                if not isinstance(item, dict):
                    missing_symbols.append(symbol)
                    continue
                ts = float(item.get("timestamp", 0.0) or 0.0)
                age = now - ts if ts > 0 else 9_999.0
                if age > self._max_age_seconds:
                    stale_symbols.append(symbol)

            price_ok = not missing_symbols and not stale_symbols
            # REST fallback keeps the feed tradable even if the websocket transport is reconnecting.
            healthy = bool(api_ok and price_ok)

            heartbeat = {
                "exchange": "kucoin",
                "api_reachable": bool(api_ok),
                "ws_active": bool(ws_ok),
                "price_recent": bool(price_ok),
                "transport_degraded": bool(api_ok and price_ok and not ws_ok),
                "missing_symbols": missing_symbols,
                "stale_symbols": stale_symbols,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            }
            self._latest_heartbeat = dict(heartbeat)
            self._healthy = bool(healthy)
            logger.bind(event="MARKET_HEARTBEAT", **heartbeat).info("MARKET_HEARTBEAT")

            if not healthy and not self._halt_active:
                reason = "market_data_unhealthy"
                self._on_halt(reason, heartbeat)
                self._halt_active = True
            elif healthy and self._halt_active:
                self._halt_active = False
                if self._on_recover is not None:
                    self._on_recover(heartbeat)

            self._stop.wait(self._interval_seconds)
