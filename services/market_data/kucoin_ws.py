from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from threading import Event, Thread
from typing import Any, Callable, Dict, Iterable, Optional

from loguru import logger
from websockets.client import connect

from .kucoin_rest import KuCoinRestClient

TickerCallback = Callable[[Dict[str, Any]], None]


class KuCoinTickerWebSocket:
    """Streams KuCoin ticker updates and forwards normalized payloads."""

    def __init__(
        self,
        symbols: Iterable[str],
        rest_client: KuCoinRestClient,
        on_ticker: TickerCallback,
    ) -> None:
        self._symbols = sorted({str(sym).upper() for sym in symbols if str(sym).strip()})
        self._rest = rest_client
        self._on_ticker = on_ticker
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self.last_message_unix: float = 0.0
        self.connected: bool = False
        self._recv_timeout_seconds: float = 15.0

    @property
    def is_active(self) -> bool:
        return bool(self.connected and (time.time() - self.last_message_unix) <= 5.0)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run_forever, name="kucoin-ticker-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_forever(self) -> None:
        asyncio.run(self._runner())

    async def _runner(self) -> None:
        while not self._stop.is_set():
            try:
                endpoint, token = self._rest.fetch_public_ws_endpoint()
                ws_url = f"{endpoint}?token={token}&connectId={uuid.uuid4().hex}"
                logger.bind(
                    event="MARKET_SOURCE",
                    exchange="kucoin",
                    source="kucoin_ws",
                    status="connecting",
                    symbols=self._symbols,
                ).info("MARKET_SOURCE")

                async with connect(ws_url, ping_interval=None, close_timeout=2.0) as ws:
                    self.connected = True
                    await self._subscribe(ws)
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    try:
                        while not self._stop.is_set():
                            raw = await asyncio.wait_for(ws.recv(), timeout=self._recv_timeout_seconds)
                            self.last_message_unix = time.time()
                            self._handle_ws_message(raw)
                    finally:
                        ping_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await ping_task
            except asyncio.TimeoutError:
                # Receive timeout; reconnect.
                pass
            except Exception as exc:
                logger.bind(
                    event="MARKET_SOURCE",
                    exchange="kucoin",
                    source="kucoin_ws",
                    status="disconnected",
                    error=str(exc),
                ).warning("MARKET_SOURCE")
            finally:
                self.connected = False

            await asyncio.sleep(1.0)

    async def _subscribe(self, ws) -> None:
        for symbol in self._symbols:
            msg = {
                "id": uuid.uuid4().hex,
                "type": "subscribe",
                "topic": f"/market/ticker:{symbol}",
                "privateChannel": False,
                "response": True,
            }
            await ws.send(json.dumps(msg))

    async def _ping_loop(self, ws) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(15.0)
            ping = {
                "id": uuid.uuid4().hex,
                "type": "ping",
            }
            await ws.send(json.dumps(ping))

    def _handle_ws_message(self, raw: Any) -> None:
        try:
            payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", errors="ignore"))
        except Exception:
            return

        if not isinstance(payload, dict):
            return
        if str(payload.get("type")) != "message":
            return

        data = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
        symbol = str(data.get("symbol", "")).upper()
        if not symbol:
            return

        ts_raw = data.get("time") if data.get("time") is not None else data.get("ts")
        ts = float(ts_raw or int(time.time() * 1000))
        ts = ts / 1000.0 if ts > 10_000_000_000 else ts

        price = float(data.get("price", 0.0) or 0.0)
        best_bid = float(data.get("bestBid", price) or price)
        best_ask = float(data.get("bestAsk", price) or price)
        volume = float(data.get("size", 0.0) or 0.0)

        if price <= 0:
            return

        self._on_ticker(
            {
                "exchange": "kucoin",
                "source": "kucoin_ws",
                "symbol": symbol,
                "price": price,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "volume": volume,
                "timestamp": ts,
            }
        )
