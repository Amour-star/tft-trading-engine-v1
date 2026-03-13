"""Run dedicated market-data service and publish validated KuCoin prices to Redis."""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List

import redis
from loguru import logger

from config.settings import ACTIVE_SYMBOL, settings
from services.market_data import HeartbeatMonitor, KuCoinRestClient, KuCoinTickerWebSocket, PriceValidationError, PriceValidator
from utils.logging import setup_logging


def _env_symbols() -> List[str]:
    raw = os.getenv("MARKET_DATA_SYMBOLS")
    if raw is None or not str(raw).strip():
        logger.error(
            "MARKET_DATA_SYMBOLS is not set. Falling back to default symbols: "
            "BTC-USDT,ETH-USDT,XRP-USDT,DOGE-USDT"
        )
        raw = "BTC-USDT,ETH-USDT,XRP-USDT,DOGE-USDT"

    parsed: List[str] = []
    for token in str(raw).split(","):
        symbol = token.strip().upper().replace("/", "-")
        if not symbol:
            continue
        if "-" not in symbol:
            symbol = f"{symbol}-USDT"
        parsed.append(symbol)

    deduped: List[str] = []
    seen = set()
    for symbol in parsed:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    if not deduped:
        logger.error("No market symbols configured from MARKET_DATA_SYMBOLS='{}'", raw)
        raise RuntimeError(
            "No market symbols configured. Market data service cannot start without symbols."
        )
    return deduped


class MarketDataService:
    def __init__(self, symbols: Iterable[str]) -> None:
        self._symbols = sorted({str(sym).upper() for sym in symbols if str(sym).strip()})
        if not self._symbols:
            raise RuntimeError(
                "MARKET_DATA_SYMBOLS environment variable not set. "
                "Market data service cannot start without symbols."
            )
        self._stop = threading.Event()
        self._exchange = "kucoin"

        self._ticker_channel = os.getenv("MARKET_TICKER_CHANNEL", "market:ticker").strip() or "market:ticker"
        self._ticker_key_prefix = os.getenv("MARKET_TICKER_KEY_PREFIX", "market:ticker:").strip() or "market:ticker:"
        self._halt_key_prefix = os.getenv("MARKET_HALT_KEY_PREFIX", "market:halt:").strip() or "market:halt:"
        self._control_channel = os.getenv("MARKET_CONTROL_CHANNEL", "market:control").strip() or "market:control"
        self._health_key = os.getenv("MARKET_HEALTH_KEY", "market:health:market-data").strip() or "market:health:market-data"
        self._health_host = os.getenv("MARKET_DATA_HEALTH_HOST", "0.0.0.0").strip() or "0.0.0.0"
        try:
            self._health_port = int(os.getenv("MARKET_DATA_HEALTH_PORT", "8010") or "8010")
        except ValueError:
            self._health_port = 8010
        self._health_server: ThreadingHTTPServer | None = None
        self._health_thread: threading.Thread | None = None
        self._recovery_interval_seconds = 5.0
        self._last_recovery_attempt = 0.0

        self._redis = redis.Redis(
            host=settings.redis.host,
            port=int(settings.redis.port),
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )

        self._validator = PriceValidator()
        self._rest = KuCoinRestClient(
            base_url=settings.kucoin.base_url,
            timeout_seconds=float(getattr(settings.kucoin, "http_timeout_seconds", 4.0)),
        )
        self._latest_by_symbol: Dict[str, Dict[str, Any]] = {}

        self._ws = KuCoinTickerWebSocket(
            symbols=self._symbols,
            rest_client=self._rest,
            on_ticker=self._on_ticker,
        )
        self._heartbeat = HeartbeatMonitor(
            symbols=self._symbols,
            rest_client=self._rest,
            ws_active_provider=lambda: self._ws.is_active,
            latest_ticker_provider=self._ticker_snapshot,
            on_halt=self._on_halt,
            on_recover=self._on_recover,
            interval_seconds=2.0,
            max_age_seconds=float(getattr(settings.kucoin, "market_data_max_age_seconds", 30.0)),
        )

    def _prime_symbols_from_rest(self, symbols: Iterable[str], reason: str) -> None:
        for symbol in sorted({str(item).upper() for item in symbols if str(item).strip()}):
            try:
                payload = self._rest.fetch_level1_ticker(symbol)
                payload["source"] = "kucoin_rest"
                self._on_ticker(payload)
                logger.bind(
                    event="MARKET_SOURCE",
                    exchange=self._exchange,
                    source="kucoin_rest",
                    symbol=symbol,
                    reason=reason,
                ).info("MARKET_SOURCE")
            except Exception as exc:
                logger.bind(
                    event="MARKET_SOURCE",
                    exchange=self._exchange,
                    source="kucoin_rest",
                    symbol=symbol,
                    reason=reason,
                    error=str(exc),
                ).warning("MARKET_SOURCE")

    def _start_health_server(self) -> None:
        service = self

        class _HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/health/market-data":
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.end_headers()
                    return
                payload = service.health_snapshot()
                body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

        self._health_server = ThreadingHTTPServer((self._health_host, self._health_port), _HealthHandler)
        self._health_thread = threading.Thread(
            target=self._health_server.serve_forever,
            name="market-data-health",
            daemon=True,
        )
        self._health_thread.start()

    def _stop_health_server(self) -> None:
        if self._health_server is not None:
            try:
                self._health_server.shutdown()
                self._health_server.server_close()
            except Exception:
                logger.exception("Failed to shutdown market-data health server cleanly")
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=2.0)
        self._health_server = None
        self._health_thread = None

    def _ticker_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {symbol: dict(value) for symbol, value in self._latest_by_symbol.items()}

    def health_snapshot(self) -> Dict[str, Any]:
        heartbeat = self._heartbeat.latest_heartbeat
        now = time.time()
        stale = []
        missing = []
        for symbol in self._symbols:
            payload = self._latest_by_symbol.get(symbol)
            if not isinstance(payload, dict):
                missing.append(symbol)
                continue
            ts = float(payload.get("timestamp", 0.0) or 0.0)
            if ts <= 0 or (now - ts) > self._market_data_max_age_seconds:
                stale.append(symbol)
        prices_recent = not missing and not stale
        return {
            "exchange": self._exchange,
            "ws_active": bool(self._ws.is_active),
            "symbols": list(self._symbols),
            "prices_recent": bool(prices_recent),
            "missing_symbols": sorted(set(missing + stale)),
            "timestamp": datetime.utcnow().isoformat(),
            "heartbeat": heartbeat,
        }

    def _persist_health_snapshot(self) -> None:
        try:
            snapshot = self.health_snapshot()
            payload = json.dumps(snapshot, separators=(",", ":"), sort_keys=True)
            self._redis.set(self._health_key, payload)
        except Exception as exc:
            logger.debug("Failed to persist market-data health snapshot: {}", exc)

    def _attempt_recovery(self, heartbeat: Dict[str, Any]) -> None:
        now = time.time()
        if (now - self._last_recovery_attempt) < self._recovery_interval_seconds:
            return
        self._last_recovery_attempt = now
        impacted = heartbeat.get("missing_symbols") or heartbeat.get("stale_symbols") or self._symbols
        self._prime_symbols_from_rest(impacted, reason="recovery")
        logger.bind(
            event="MARKET_RECOVERY",
            exchange=self._exchange,
            reason="heartbeat_unhealthy",
            heartbeat=heartbeat,
        ).warning("MARKET_RECOVERY")
        try:
            self._ws.stop()
        except Exception:
            logger.exception("Failed to stop websocket during recovery")
        try:
            self._ws.start()
        except Exception:
            logger.exception("Failed to restart websocket during recovery")

    def _on_ticker(self, payload: Dict[str, Any]) -> None:
        try:
            validated = self._validator.validate(payload)
        except PriceValidationError:
            return

        symbol = str(validated.get("symbol")).upper()
        self._latest_by_symbol[symbol] = dict(validated)
        self._publish_ticker(validated)

    def _publish_ticker(self, ticker: Dict[str, Any]) -> None:
        symbol = str(ticker.get("symbol")).upper()
        body = dict(ticker)
        body["last_update"] = datetime.utcnow().isoformat()

        payload = json.dumps(body, separators=(",", ":"), sort_keys=True)
        self._redis.set(f"{self._ticker_key_prefix}{symbol}", payload)
        self._redis.publish(self._ticker_channel, payload)
        self._redis.delete(f"{self._halt_key_prefix}{symbol}")
        self._persist_health_snapshot()

        logger.bind(
            event="MARKET_SOURCE",
            exchange=str(body.get("exchange") or "kucoin"),
            source=str(body.get("source") or "kucoin_ws"),
            symbol=symbol,
            latency_ms=float(body.get("latency_ms") or 0.0),
            last_update=body["last_update"],
            channel=self._ticker_channel,
        ).info("MARKET_SOURCE")

    def _on_halt(self, reason: str, heartbeat: Dict[str, Any]) -> None:
        now_iso = datetime.utcnow().isoformat()
        impacted = sorted(
            {
                *[str(item).upper() for item in heartbeat.get("missing_symbols", [])],
                *[str(item).upper() for item in heartbeat.get("stale_symbols", [])],
            }
        )
        if not impacted:
            impacted = list(self._symbols)

        for symbol in impacted:
            halt_payload = {
                "event": "TRADING_HALTED",
                "symbol": symbol,
                "reason": str(reason),
                "timestamp": now_iso,
                "details": heartbeat,
            }
            self._redis.set(f"{self._halt_key_prefix}{symbol}", json.dumps(halt_payload))
            self._redis.publish(self._control_channel, json.dumps(halt_payload))
        self._persist_health_snapshot()

        logger.bind(
            event="TRADING_HALTED",
            exchange="kucoin",
            reason=str(reason),
            impacted_symbols=impacted,
            details=heartbeat,
        ).critical("TRADING_HALTED")

    def _on_recover(self, heartbeat: Dict[str, Any]) -> None:
        for symbol in self._symbols:
            self._redis.delete(f"{self._halt_key_prefix}{symbol}")
        self._persist_health_snapshot()
        logger.bind(
            event="MARKET_HEARTBEAT",
            exchange="kucoin",
            status="recovered",
            details=heartbeat,
        ).warning("MARKET_HEARTBEAT")

    def run(self) -> None:
        self._redis.ping()
        try:
            self._start_health_server()
        except Exception as exc:
            logger.error("Failed to start market-data health endpoint: {}", exc)
        logger.info("MarketDataService started")
        logger.info("Symbols: {}", ",".join(self._symbols))
        logger.info("Exchange: {}", self._exchange)
        logger.info("Redis: connected (host={} port={})", settings.redis.host, settings.redis.port)
        logger.info("Subscribed channels: {}", ",".join([self._ticker_channel, self._control_channel]))
        self._prime_symbols_from_rest(self._symbols, reason="startup")
        self._ws.start()
        self._heartbeat.start()
        self._persist_health_snapshot()
        time.sleep(1.0)
        logger.info("WebSocket: {}", "connected" if self._ws.connected else "disconnected")

        while not self._stop.wait(1.0):
            self._persist_health_snapshot()
            heartbeat = self._heartbeat.latest_heartbeat
            if heartbeat and not self._heartbeat.is_healthy:
                self._attempt_recovery(heartbeat)

    def stop(self) -> None:
        self._stop.set()
        self._heartbeat.stop()
        self._ws.stop()
        self._stop_health_server()


def main() -> None:
    setup_logging()
    try:
        symbols = _env_symbols()
        service = MarketDataService(symbols)
    except Exception as exc:
        logger.error(
            "Market data service startup failed: {}. Set MARKET_DATA_SYMBOLS (e.g. BTC-USDT,ETH-USDT,XRP-USDT,DOGE-USDT).",
            exc,
        )
        raise

    def _handle_signal(signum, _frame) -> None:  # type: ignore[no-untyped-def]
        logger.warning("Market-data service received signal {}", signum)
        service.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        service.run()
    except Exception as exc:
        logger.bind(event="TRADING_HALTED", reason=str(exc)).critical("TRADING_HALTED")
        raise
    finally:
        service.stop()


if __name__ == "__main__":
    main()
