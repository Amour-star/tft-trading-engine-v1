from __future__ import annotations

import time
from typing import Dict, Tuple

import requests


class KuCoinRestClient:
    """Thin REST helper for KuCoin connectivity and WS bootstrap."""

    def __init__(self, base_url: str = "https://api.kucoin.com", timeout_seconds: float = 4.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def is_reachable(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/timestamp",
                timeout=self.timeout_seconds,
            )
            if response.status_code != 200:
                return False
            payload = response.json()
            return str(payload.get("code", "")) in {"200000", "200"}
        except Exception:
            return False

    def fetch_public_ws_endpoint(self) -> Tuple[str, str]:
        response = requests.post(
            f"{self.base_url}/api/v1/bullet-public",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        if str(payload.get("code", "")) not in {"200000", "200"}:
            raise RuntimeError(f"Invalid KuCoin bullet response: {payload}")

        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        token = str(data.get("token", ""))
        servers = data.get("instanceServers", []) if isinstance(data, dict) else []
        endpoint = ""
        if isinstance(servers, list) and servers:
            endpoint = str((servers[0] or {}).get("endpoint", ""))

        if not token or not endpoint:
            raise RuntimeError("KuCoin WS token/endpoint unavailable")
        return endpoint, token

    def fetch_level1_ticker(self, symbol: str) -> Dict[str, float]:
        response = requests.get(
            f"{self.base_url}/api/v1/market/orderbook/level1",
            params={"symbol": symbol},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        if str(payload.get("code", "")) not in {"200000", "200"}:
            raise RuntimeError(f"Invalid KuCoin ticker response: {payload}")
        data = payload.get("data", {}) if isinstance(payload, dict) else {}

        now = time.time()
        price = float(data.get("price", 0.0) or 0.0)
        best_bid = float(data.get("bestBid", price) or price)
        best_ask = float(data.get("bestAsk", price) or price)
        size = float(data.get("size", 0.0) or 0.0)
        return {
            "symbol": symbol,
            "price": price,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "volume": size if size > 0 else 1e-9,
            # REST snapshots should be considered fresh at fetch time even if the last
            # matched trade timestamp is older for a quieter symbol.
            "timestamp": now,
            "exchange": "kucoin",
            "source": "kucoin_rest",
        }
