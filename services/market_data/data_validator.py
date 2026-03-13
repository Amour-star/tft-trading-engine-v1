from __future__ import annotations

import time
from typing import Any, Dict

from loguru import logger


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_unix_seconds(raw_ts: Any) -> float:
    ts = _as_float(raw_ts, 0.0)
    if ts <= 0:
        return 0.0
    # Most exchange feeds use milliseconds.
    if ts > 10_000_000_000:
        return ts / 1000.0
    return ts


class PriceValidationError(RuntimeError):
    """Raised when incoming live price fails validation."""


class PriceValidator:
    """Validates ticker quality before publishing to trading engines."""

    def __init__(self) -> None:
        self._last_price_by_symbol: Dict[str, float] = {}

    def validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(payload.get("symbol", "")).upper()
        if not symbol:
            raise PriceValidationError("Missing symbol")

        price = _as_float(payload.get("price"), 0.0)
        best_bid = _as_float(payload.get("best_bid"), price)
        best_ask = _as_float(payload.get("best_ask"), price)
        volume = _as_float(payload.get("volume", payload.get("size")), 0.0)
        ts = _to_unix_seconds(payload.get("timestamp", payload.get("time")))
        now = time.time()

        if price <= 0:
            self._log_fail(symbol, "non_positive_price", payload)
            raise PriceValidationError("non_positive_price")

        if volume <= 0:
            self._log_fail(symbol, "zero_volume", payload)
            raise PriceValidationError("zero_volume")

        if ts <= 0:
            self._log_fail(symbol, "missing_timestamp", payload)
            raise PriceValidationError("missing_timestamp")

        age = now - ts
        if age > 5.0:
            self._log_fail(symbol, f"stale_timestamp:{age:.3f}s", payload)
            raise PriceValidationError("stale_timestamp")

        last_price = self._last_price_by_symbol.get(symbol)
        if last_price is not None and last_price > 0:
            drift = abs(price - last_price) / last_price
            if drift > 0.03:
                self._log_fail(symbol, f"price_drift:{drift:.4f}", payload)
                raise PriceValidationError("price_drift")

        self._last_price_by_symbol[symbol] = price

        if best_bid <= 0:
            best_bid = price
        if best_ask <= 0:
            best_ask = price

        return {
            "exchange": str(payload.get("exchange") or "kucoin"),
            "source": str(payload.get("source") or "kucoin_ws"),
            "symbol": symbol,
            "price": float(price),
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "volume": float(volume),
            "timestamp": float(ts),
            "latency_ms": float(max(0.0, (now - ts) * 1000.0)),
            "last_update": payload.get("last_update") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }

    @staticmethod
    def _log_fail(symbol: str, reason: str, payload: Dict[str, Any]) -> None:
        logger.bind(
            event="PRICE_VALIDATION_FAIL",
            symbol=symbol,
            reason=reason,
            source=str(payload.get("source") or "unknown"),
            exchange=str(payload.get("exchange") or "kucoin"),
        ).error("PRICE_VALIDATION_FAIL")
