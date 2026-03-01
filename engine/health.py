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
                if self.path != "/health":
                    self.send_response(404)
                    self.end_headers()
                    return

                payload = status_provider()
                ready = bool(payload.get("ready", True))
                status_code = 200 if ready else 503
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