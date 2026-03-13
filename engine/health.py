"""
Lightweight healthcheck HTTP server for the engine process.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict


class HealthServer:
    def __init__(self, host: str, port: int, status_provider: Callable[[], Dict[str, object]]) -> None:
        self._host = host
        self._port = port
        self._status_provider = status_provider
        self._server = ThreadingHTTPServer((self._host, self._port), self._make_handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="engine-health",
            daemon=True,
        )

    def _make_handler(self):
        status_provider = self._status_provider

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
                path = str(self.path or "").split("?", 1)[0]

                if path not in {"/health", "/market/status", "/market/source", "/market/latency"}:
                    self.send_response(404)
                    self.end_headers()
                    return

                status = status_provider()
                market_payload = {
                    "exchange": status.get("exchange", "kucoin"),
                    "source": status.get("source", status.get("ticker_source", "market_data_service")),
                    "latency_ms": status.get("latency_ms"),
                    "last_update": status.get("last_update", status.get("data_age_seconds")),
                    "market_data_ready": bool(status.get("market_data_ready", False)),
                    "market_data_source": status.get("market_data_source", "market_data_service"),
                    "ticker_source": status.get("ticker_source", "unknown"),
                    "orderbook_source": status.get("orderbook_source", "unknown"),
                    "warning": status.get("market_data_warning", ""),
                }

                if path == "/health":
                    payload = status
                    ready = bool(payload.get("ready", True))
                    status_code = 200 if ready else 503
                elif path == "/market/status":
                    payload = market_payload
                    status_code = 200 if market_payload["market_data_ready"] else 503
                elif path == "/market/source":
                    payload = {
                        "exchange": market_payload["exchange"],
                        "source": market_payload["source"],
                        "market_data_source": market_payload["market_data_source"],
                        "ticker_source": market_payload["ticker_source"],
                        "orderbook_source": market_payload["orderbook_source"],
                        "last_update": market_payload["last_update"],
                    }
                    status_code = 200
                else:
                    payload = {
                        "exchange": market_payload["exchange"],
                        "source": market_payload["source"],
                        "latency_ms": market_payload["latency_ms"],
                        "last_update": market_payload["last_update"],
                    }
                    status_code = 200

                body = json.dumps(payload).encode("utf-8")

                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                return

        return Handler

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        try:
            self._server.shutdown()
        finally:
            self._server.server_close()
