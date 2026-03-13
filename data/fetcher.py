"""
KuCoin market data fetcher.
Handles OHLCV retrieval, top pairs discovery, and strict live ticker access.
"""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import redis
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from requests import exceptions as requests_exceptions

from config.settings import ACTIVE_SYMBOL, XRP_ONLY_SYMBOL, settings

try:
    from kucoin.client import Market, Trade, User
except Exception as exc:
    Market = Trade = User = None  # type: ignore[assignment]
    logger.warning(f"kucoin-python unavailable ({exc}). Using offline-safe mode.")


TIMEFRAME_MAP = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1hour": "1hour",
    "4hour": "4hour",
    "1day": "1day",
}

TIMEFRAME_SECONDS = {
    "1min": 60,
    "5min": 300,
    "15min": 900,
    "1hour": 3600,
    "4hour": 14400,
    "1day": 86400,
}

_FALLBACK_PAIRS = [
    {"symbol": ACTIVE_SYMBOL, "volValue": "180000000"},
]

_FALLBACK_SYMBOL_INFO = {
    "baseMinSize": "0.00001",
    "baseMaxSize": "1000000",
    "baseIncrement": "0.00001",
    "priceIncrement": "0.00001",
    "quoteMinSize": "0.01",
    "quoteIncrement": "0.01",
}

_AUTH_ERROR_CODES = {"400001", "400002", "400003", "400004", "401", "401000"}
_RATE_LIMIT_CODES = {"429000", "429001", "429002", "429"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_placeholder_secret(value: str) -> bool:
    probe = str(value or "").strip().lower()
    if not probe:
        return True
    return probe.startswith("your_") or probe in {"changeme", "change_me", "replace_me"}


def _offline_mode_enabled() -> bool:
    raw = os.getenv("KUCOIN_OFFLINE_MODE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _normalize_error_message(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_auth_error(status_code: int, payload: Optional[Dict[str, Any]], message: str) -> bool:
    normalized = _normalize_error_message(message)
    code = _normalize_error_message((payload or {}).get("code"))
    if status_code in {401, 403}:
        return True
    if code in _AUTH_ERROR_CODES:
        return True
    if "kc-api-key not exists" in normalized:
        return True
    if "invalid kc-api-key" in normalized:
        return True
    if "invalid api key" in normalized:
        return True
    if "api key" in normalized and "not exist" in normalized:
        return True
    if "signature" in normalized and "invalid" in normalized:
        return True
    return False


def _is_rate_limited(status_code: int, payload: Optional[Dict[str, Any]], message: str) -> bool:
    normalized = _normalize_error_message(message)
    code = _normalize_error_message((payload or {}).get("code"))
    if status_code == 429:
        return True
    if code in _RATE_LIMIT_CODES:
        return True
    return "rate limit" in normalized or "too many requests" in normalized


def _parse_universe_from_env(default_symbol: str) -> List[str]:
    raw = os.getenv("UNIVERSE", "").strip()
    if not raw:
        return [default_symbol]
    symbols = []
    for item in raw.split(","):
        sym = item.strip().upper().replace("/", "-")
        if not sym:
            continue
        if "-" not in sym:
            sym = f"{sym}-USDT"
        symbols.append(sym)
    if not symbols:
        return [default_symbol]
    deduped: List[str] = []
    seen = set()
    for sym in symbols:
        if sym in seen:
            continue
        seen.add(sym)
        deduped.append(sym)
    return deduped


class KuCoinDataFetcher:
    """Fetches and manages market data from KuCoin."""

    def __init__(self) -> None:
        cfg = settings.kucoin
        url = cfg.base_url
        self._config = cfg
        self._offline_mode = _offline_mode_enabled() or Market is None
        # Synthetic market-data fallback is intentionally disabled.
        self._allow_synthetic_orderbook = False
        self._allow_synthetic_ticker = False
        self._allow_synthetic_klines = False
        self._orderbook_retry_attempts = max(1, int(getattr(cfg, "orderbook_retry_attempts", 3)))
        self._http_retry_attempts = max(1, int(getattr(cfg, "http_retry_attempts", 4)))
        self._http_timeout_seconds = max(1.0, float(getattr(cfg, "http_timeout_seconds", 4.0)))
        self._http_max_backoff_seconds = max(1.0, float(getattr(cfg, "http_max_backoff_seconds", 8.0)))
        self._market_data_max_age_seconds = max(
            5.0,
            float(getattr(cfg, "market_data_max_age_seconds", 30.0)),
        )

        self.market = None
        self.trade_client = None
        self.user_client = None
        self._public_only_mode = False
        self._credentials_present = False
        self._auth_required = bool(getattr(cfg, "require_auth", False))
        self._auth_valid: Optional[bool] = None
        self._auth_error = ""
        self._startup_checked = False
        self._status_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_prices: Dict[str, float] = {
            str(ACTIVE_SYMBOL).upper(): 0.6,
        }
        self._last_valid_ticker: Dict[str, Dict[str, Any]] = {}
        self._market_exchange = "kucoin"
        self._ticker_channel = os.getenv("MARKET_TICKER_CHANNEL", "market:ticker").strip() or "market:ticker"
        self._ticker_key_prefix = (
            os.getenv("MARKET_TICKER_KEY_PREFIX", "market:ticker:").strip() or "market:ticker:"
        )
        self._halt_key_prefix = (
            os.getenv("MARKET_HALT_KEY_PREFIX", "market:halt:").strip() or "market:halt:"
        )
        self._redis: Optional[redis.Redis] = None
        self._pubsub = None
        self._redis_connected = False
        self._redis_error: str = ""

        if not self._offline_mode:
            try:
                self.market = Market(url=url) if Market else None
                self._credentials_present = (
                    not _is_placeholder_secret(cfg.api_key)
                    and not _is_placeholder_secret(cfg.api_secret)
                    and not _is_placeholder_secret(cfg.api_passphrase)
                )
                self.trade_client = (
                    Trade(
                        key=cfg.api_key,
                        secret=cfg.api_secret,
                        passphrase=cfg.api_passphrase,
                        url=url,
                    )
                    if Trade and self._credentials_present
                    else None
                )
                self.user_client = (
                    User(
                        key=cfg.api_key,
                        secret=cfg.api_secret,
                        passphrase=cfg.api_passphrase,
                        url=url,
                    )
                    if User and self._credentials_present
                    else None
                )
            except Exception as exc:
                logger.warning(f"Failed to initialize KuCoin clients, using offline mode: {exc}")
                self._offline_mode = True

        self._init_market_data_bus()

        self._refresh_status(
            ACTIVE_SYMBOL,
            ticker_source="startup",
            orderbook_source="startup",
            market_data_source="market_data_service",
        )
        self._run_startup_auth_check(force=True)

    def _init_market_data_bus(self) -> None:
        try:
            self._redis = redis.Redis(
                host=settings.redis.host,
                port=int(settings.redis.port),
                decode_responses=True,
                socket_connect_timeout=1.5,
                socket_timeout=1.5,
            )
            self._redis.ping()
            self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
            self._pubsub.subscribe(self._ticker_channel)
            self._redis_connected = True
            self._redis_error = ""
            logger.bind(
                event="MARKET_SOURCE",
                exchange=self._market_exchange,
                source="market_data_service",
                channel=self._ticker_channel,
                status="connected",
            ).info("MARKET_SOURCE")
        except Exception as exc:
            self._redis_connected = False
            self._redis_error = str(exc)
            self._pubsub = None
            logger.bind(
                event="MARKET_SOURCE",
                exchange=self._market_exchange,
                source="market_data_service",
                channel=self._ticker_channel,
                status="unavailable",
                error=self._redis_error,
            ).error("MARKET_SOURCE")

    @staticmethod
    def _ticker_timestamp_to_seconds(value: Any) -> float:
        raw = _as_float(value, 0.0)
        if raw <= 0:
            return 0.0
        # KuCoin WS usually emits milliseconds.
        if raw > 10_000_000_000:
            return raw / 1000.0
        return raw

    def _ingest_market_ticker(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        symbol = str(payload.get("symbol", "")).upper()
        if not symbol:
            return None

        price = _as_float(payload.get("price"), 0.0)
        if price <= 0:
            return None
        best_bid = _as_float(payload.get("best_bid"), price)
        best_ask = _as_float(payload.get("best_ask"), price)
        if best_bid <= 0:
            best_bid = price
        if best_ask <= 0:
            best_ask = price

        timestamp_raw = (
            payload.get("timestamp")
            if payload.get("timestamp") is not None
            else payload.get("time")
        )
        timestamp_sec = self._ticker_timestamp_to_seconds(timestamp_raw)
        if timestamp_sec <= 0:
            timestamp_sec = time.time()
        now = time.time()
        latency_ms = max(0.0, (now - timestamp_sec) * 1000.0)

        normalized = {
            "symbol": symbol,
            "exchange": str(payload.get("exchange") or self._market_exchange),
            "source": str(payload.get("source") or "market_data_service"),
            "price": float(price),
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "size": _as_float(payload.get("volume", payload.get("size")), 0.0),
            "time": float(timestamp_sec * 1000.0),
            "latency_ms": float(latency_ms),
            "received_at_unix": float(now),
            "updated_at": datetime.utcnow().isoformat(),
        }

        self._last_prices[symbol] = float(price)
        self._last_valid_ticker[symbol] = dict(normalized)
        self._refresh_status(
            symbol,
            ticker_source="market_data_service",
            market_data_source="market_data_service",
        )
        return normalized

    def _read_market_ticker_from_key(self, symbol: str) -> Optional[Dict[str, Any]]:
        if self._redis is None:
            return None
        key = f"{self._ticker_key_prefix}{str(symbol).upper()}"
        raw = self._redis.get(key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        parsed = json.loads(str(raw))
        if not isinstance(parsed, dict):
            return None
        return self._ingest_market_ticker(parsed)

    def _drain_market_ticker_channel(self) -> None:
        if self._pubsub is None:
            return
        for _ in range(16):
            msg = self._pubsub.get_message(timeout=0.01)
            if not msg:
                break
            if str(msg.get("type")) != "message":
                continue
            payload_raw = msg.get("data")
            try:
                if isinstance(payload_raw, bytes):
                    payload_raw = payload_raw.decode("utf-8", errors="ignore")
                parsed = json.loads(str(payload_raw))
                if isinstance(parsed, dict):
                    self._ingest_market_ticker(parsed)
            except Exception as exc:
                logger.debug("Failed to parse market:ticker message: {}", exc)

    def _is_market_halted(self, symbol: str) -> bool:
        if self._redis is None:
            return False
        try:
            value = self._redis.get(f"{self._halt_key_prefix}{str(symbol).upper()}")
            return bool(value)
        except Exception:
            return False

    def _resolve_live_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        symbol_key = str(symbol or ACTIVE_SYMBOL).upper()
        if not self._redis_connected:
            self._init_market_data_bus()
        self._drain_market_ticker_channel()
        cached = self._last_valid_ticker.get(symbol_key)
        if cached:
            age = time.time() - _as_float(cached.get("received_at_unix"), 0.0)
            if 0.0 <= age <= self._market_data_max_age_seconds:
                return {
                    "symbol": symbol_key,
                    "price": _as_float(cached.get("price"), 0.0),
                    "best_bid": _as_float(cached.get("best_bid"), 0.0),
                    "best_ask": _as_float(cached.get("best_ask"), 0.0),
                    "size": _as_float(cached.get("size"), 0.0),
                    "time": _as_float(cached.get("time"), float(int(time.time() * 1000))),
                    "source": str(cached.get("source") or "market_data_service"),
                    "latency_ms": _as_float(cached.get("latency_ms"), 0.0),
                    "exchange": str(cached.get("exchange") or self._market_exchange),
                }

        try:
            return self._read_market_ticker_from_key(symbol_key)
        except Exception as exc:
            self._redis_error = str(exc)
            logger.debug("Market ticker key lookup failed for {}: {}", symbol_key, exc)
            return None

    @staticmethod
    def _assert_xrp_symbol(symbol: str) -> None:
        # Now allows any symbol assigned to this engine via SYMBOL env var
        pass

    def _fallback_pairs(self, top_n: int) -> List[Dict[str, Any]]:
        universe = _parse_universe_from_env(ACTIVE_SYMBOL)
        if not universe:
            return list(_FALLBACK_PAIRS)
        fallback = []
        base_volume = 200_000_000.0
        for idx, symbol in enumerate(universe[: max(1, int(top_n))]):
            fallback.append(
                {
                    "symbol": symbol,
                    "volValue": str(max(1.0, base_volume - idx * 10_000_000.0)),
                }
            )
        return fallback

    def _fallback_symbol_info(self, symbol: str) -> Dict[str, Any]:
        self._assert_xrp_symbol(symbol)
        base, quote = symbol.split("-") if "-" in symbol else (symbol, "USDT")
        return {
            "symbol": symbol,
            "base_currency": base,
            "quote_currency": quote,
            "base_min_size": _as_float(_FALLBACK_SYMBOL_INFO["baseMinSize"], 0.00001),
            "base_max_size": _as_float(_FALLBACK_SYMBOL_INFO["baseMaxSize"], 1_000_000.0),
            "base_increment": _as_float(_FALLBACK_SYMBOL_INFO["baseIncrement"], 0.00001),
            "price_increment": _as_float(_FALLBACK_SYMBOL_INFO["priceIncrement"], 0.00001),
            "quote_min_size": _as_float(_FALLBACK_SYMBOL_INFO["quoteMinSize"], 0.01),
            "quote_increment": _as_float(_FALLBACK_SYMBOL_INFO["quoteIncrement"], 0.01),
            "fee_currency": quote,
        }

    def _base_price(self, symbol: str) -> float:
        self._assert_xrp_symbol(symbol)
        key = str(symbol or ACTIVE_SYMBOL).upper()
        if key in self._last_prices:
            return self._last_prices[key]
        base = key.split("-")[0]
        seed = sum(ord(ch) for ch in base) % 1000
        price = max(0.05, float(seed))
        self._last_prices[key] = price
        return price

    def _generate_synthetic_klines(
        self,
        symbol: str,
        timeframe: str,
        start_dt: Optional[datetime],
        end_dt: Optional[datetime],
    ) -> pd.DataFrame:
        self._assert_xrp_symbol(symbol)
        interval = TIMEFRAME_SECONDS.get(timeframe, 3600)
        end_time = end_dt or datetime.utcnow()
        start_time = start_dt or (end_time - timedelta(seconds=interval * 500))
        if start_time >= end_time:
            start_time = end_time - timedelta(seconds=interval * 100)

        n_points = max(120, int((end_time - start_time).total_seconds() // interval))
        timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
        rng_seed = abs(hash((symbol, timeframe, n_points))) % (2**32)
        rng = np.random.default_rng(rng_seed)

        base_price = self._base_price(symbol)
        returns = rng.normal(0.0, 0.002, n_points)
        close = base_price * np.exp(np.cumsum(returns))
        spread = np.maximum(close * 0.001, 1e-8)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.lognormal(mean=8.0, sigma=0.4, size=n_points)
        turnover = volume * close

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps, utc=False),
                "open": open_.astype(float),
                "close": close.astype(float),
                "high": high.astype(float),
                "low": low.astype(float),
                "volume": volume.astype(float),
                "turnover": turnover.astype(float),
            }
        )
        self._last_prices[str(symbol or ACTIVE_SYMBOL).upper()] = float(df["close"].iloc[-1])
        return df

    def _synthetic_orderbook(self, symbol: str, depth: int) -> Dict[str, Any]:
        self._assert_xrp_symbol(symbol)
        price = self._base_price(symbol)
        bids = [(price * (1.0 - i * 0.0005), 10.0 + i) for i in range(depth)]
        asks = [(price * (1.0 + i * 0.0005), 10.0 + i) for i in range(depth)]
        return {"bids": bids, "asks": asks, "time": int(time.time() * 1000), "source": "synthetic"}

    def _synthetic_ticker(self, symbol: str) -> Dict[str, Any]:
        self._assert_xrp_symbol(symbol)
        price = float(self._base_price(symbol))
        return {
            "price": price,
            "best_bid": price * 0.9995,
            "best_ask": price * 1.0005,
            "size": 1.0,
            "time": float(int(time.time() * 1000)),
            "source": "synthetic",
        }

    def _cached_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._assert_xrp_symbol(symbol)
        key = str(symbol or ACTIVE_SYMBOL).upper()
        cached = self._last_valid_ticker.get(key, {})
        price = _as_float(cached.get("price"), _as_float(self._last_prices.get(key), 0.0))
        if price <= 0:
            return None
        best_bid = _as_float(cached.get("best_bid"), max(1e-8, price * 0.9995))
        best_ask = _as_float(cached.get("best_ask"), max(best_bid, price * 1.0005))
        source_ts = _as_float(cached.get("time"), float(int(time.time() * 1000)))
        return {
            "price": price,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "size": _as_float(cached.get("size"), 0.0),
            "time": source_ts,
            "source": "last_known",
            "cached_at": str(cached.get("updated_at", "")),
        }

    def _cache_valid_ticker(self, symbol: str, ticker: Dict[str, Any]) -> None:
        key = str(symbol or ACTIVE_SYMBOL).upper()
        price = _as_float(ticker.get("price"), 0.0)
        bid = _as_float(ticker.get("best_bid"), price)
        ask = _as_float(ticker.get("best_ask"), price)
        if price <= 0 or bid <= 0 or ask <= 0:
            return
        source = str(ticker.get("source", "unknown")).lower()
        if source == "synthetic":
            return
        now_iso = datetime.utcnow().isoformat()
        self._last_prices[key] = price
        self._last_valid_ticker[key] = {
            "price": price,
            "best_bid": bid,
            "best_ask": ask,
            "size": _as_float(ticker.get("size"), 0.0),
            "time": _as_float(ticker.get("time"), float(int(time.time() * 1000))),
            "source": source,
            "updated_at": now_iso,
            "received_at_unix": time.time(),
        }

    def _get_ticker_age_seconds(self, symbol: str) -> Optional[float]:
        key = str(symbol or ACTIVE_SYMBOL).upper()
        cached = self._last_valid_ticker.get(key, {})
        received_at = _as_float(cached.get("received_at_unix"), -1.0)
        if received_at <= 0:
            return None
        return max(0.0, float(time.time() - received_at))

    @staticmethod
    def _is_live_ticker_source(source: str) -> bool:
        src = str(source or "").strip().lower()
        return src in {"market_data_service", "kucoin_ws", "kucoin_rest"}

    def _compute_market_data_health(self, symbol: str, status: Dict[str, Any]) -> Dict[str, Any]:
        key = str(symbol or ACTIVE_SYMBOL).upper()
        source = str(status.get("market_data_source", "market_data_service")).lower()
        ticker_source = str(status.get("ticker_source", "unknown")).lower()
        auth_required = bool(status.get("auth_required", False))
        auth_valid = status.get("auth_valid")
        credentials_present = bool(status.get("credentials_present", False))
        halted = self._is_market_halted(key)

        cached = self._last_valid_ticker.get(key, {})
        last_price = _as_float(cached.get("price"), _as_float(self._last_prices.get(key), 0.0))
        last_price_ts = str(cached.get("updated_at", "")) or None
        age_seconds = self._get_ticker_age_seconds(key)
        is_stale = age_seconds is None or age_seconds > self._market_data_max_age_seconds

        credentials_valid = bool(credentials_present and auth_valid is not False)
        if auth_required:
            credentials_valid = bool(auth_valid is True)

        market_data_ready = bool(
            last_price > 0.0
            and source == "market_data_service"
            and self._is_live_ticker_source(ticker_source)
            and not is_stale
            and not halted
            and self._redis_connected
        )
        trading_enabled = bool(market_data_ready and (not auth_required or credentials_valid))

        warning = ""
        if halted:
            warning = "market_data_halted"
        elif not self._redis_connected:
            warning = f"market_data_bus_unavailable:{self._redis_error or 'redis_unreachable'}"
        elif source != "market_data_service":
            warning = f"invalid_market_data_source:{source or 'unknown'}"
        elif not self._is_live_ticker_source(ticker_source):
            warning = f"ticker_source_unavailable:{ticker_source or 'unknown'}"
        elif last_price <= 0:
            warning = "invalid_or_missing_price"
        elif is_stale:
            warning = (
                f"stale_market_data:{age_seconds:.1f}s"
                if age_seconds is not None
                else "stale_market_data:unknown_age"
            )

        return {
            "last_price": float(last_price),
            "last_price_timestamp": last_price_ts,
            "data_age_seconds": float(age_seconds) if age_seconds is not None else None,
            "market_data_max_age_seconds": float(self._market_data_max_age_seconds),
            "market_data_ready": bool(market_data_ready),
            "credentials_valid": bool(credentials_valid),
            "trading_enabled": bool(trading_enabled),
            "market_data_warning": warning,
            "exchange": self._market_exchange,
            "source": "market_data_service",
            "latency_ms": _as_float(cached.get("latency_ms"), 0.0),
            "last_update": last_price_ts,
            "halted": bool(halted),
            "redis_connected": bool(self._redis_connected),
        }

    def _retry_backoff_seconds(self, attempt: int, retry_after: Optional[str] = None) -> float:
        if retry_after:
            try:
                parsed = float(retry_after)
                if parsed > 0:
                    return min(parsed, self._http_max_backoff_seconds)
            except Exception:
                pass
        base = min(0.5 * (2 ** max(0, attempt - 1)), self._http_max_backoff_seconds)
        jitter = random.uniform(0.0, min(0.25, base * 0.3))
        return min(base + jitter, self._http_max_backoff_seconds)

    def _enable_public_only_mode(self, reason: str) -> None:
        reason_text = str(reason or "unknown auth failure")
        if not self._public_only_mode:
            logger.warning("Switching KuCoin fetcher to public-only mode: {}", reason_text)
        self._public_only_mode = True
        self.trade_client = None
        self.user_client = None
        if self._credentials_present:
            self._auth_valid = False
            self._auth_error = reason_text

    def _maybe_switch_public_mode_from_error(
        self,
        *,
        status_code: int = 0,
        payload: Optional[Dict[str, Any]] = None,
        message: str = "",
        exc: Optional[Exception] = None,
    ) -> None:
        text = message
        if not text and exc is not None:
            text = str(exc)
        if _is_auth_error(status_code, payload, text):
            self._enable_public_only_mode(text or f"HTTP {status_code}")

    def _public_get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 4.0,
    ) -> Dict[str, Any]:
        timeout = max(0.5, float(timeout_sec or self._http_timeout_seconds))
        last_error: Optional[Exception] = None

        for attempt in range(1, self._http_retry_attempts + 1):
            response = None
            payload: Optional[Dict[str, Any]] = None
            message = ""
            try:
                response = requests.get(url, params=params, timeout=timeout)
                status_code = int(response.status_code)
                try:
                    parsed = response.json()
                    if isinstance(parsed, dict):
                        payload = parsed
                        message = str(payload.get("msg", ""))
                    else:
                        message = response.text or ""
                except Exception:
                    payload = None
                    message = response.text or ""

                self._maybe_switch_public_mode_from_error(
                    status_code=status_code,
                    payload=payload,
                    message=message,
                )

                if 200 <= status_code < 300:
                    if payload is None:
                        raise ValueError("Unexpected non-dict response")
                    code = str(payload.get("code", "")).strip()
                    if code and code not in {"200000", "200"}:
                        if _is_rate_limited(status_code, payload, message) and attempt < self._http_retry_attempts:
                            backoff = self._retry_backoff_seconds(
                                attempt,
                                retry_after=response.headers.get("Retry-After"),
                            )
                            time.sleep(backoff)
                            continue
                        raise RuntimeError(f"Unexpected KuCoin response code={code} msg={message}")
                    return payload

                should_retry = (
                    _is_rate_limited(status_code, payload, message)
                    or status_code >= 500
                    or status_code == 408
                )
                if should_retry and attempt < self._http_retry_attempts:
                    backoff = self._retry_backoff_seconds(
                        attempt,
                        retry_after=(response.headers.get("Retry-After") if response is not None else None),
                    )
                    time.sleep(backoff)
                    continue

                snippet = (message or "").strip()
                if snippet:
                    snippet = snippet[:200]
                raise RuntimeError(f"HTTP {status_code} for {url}: {snippet or 'request failed'}")
            except (requests_exceptions.Timeout, requests_exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt < self._http_retry_attempts:
                    backoff = self._retry_backoff_seconds(attempt)
                    time.sleep(backoff)
                    continue
                break
            except requests_exceptions.RequestException as exc:
                last_error = exc
                self._maybe_switch_public_mode_from_error(exc=exc)
                if attempt < self._http_retry_attempts:
                    backoff = self._retry_backoff_seconds(attempt)
                    time.sleep(backoff)
                    continue
                break
            except Exception as exc:
                last_error = exc
                self._maybe_switch_public_mode_from_error(exc=exc)
                if _is_rate_limited(0, payload, message) and attempt < self._http_retry_attempts:
                    backoff = self._retry_backoff_seconds(attempt)
                    time.sleep(backoff)
                    continue
                break

        if last_error is None:
            raise RuntimeError(f"Failed to fetch JSON from {url}")
        raise RuntimeError(f"Failed to fetch JSON from {url}: {last_error}") from last_error

    @staticmethod
    def _to_binance_symbol(symbol: str) -> str:
        return symbol.replace("-", "").replace("/", "").upper()

    def _fetch_ticker_via_kucoin_rest(self, symbol: str) -> Dict[str, Any]:
        payload = self._public_get_json(
            f"{settings.kucoin.base_url}/api/v1/market/orderbook/level1",
            params={"symbol": symbol},
            timeout_sec=self._http_timeout_seconds,
        )
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        price = _as_float(data.get("price"), 0.0)
        best_bid = _as_float(data.get("bestBid"), price)
        best_ask = _as_float(data.get("bestAsk"), price)
        if price <= 0:
            price = _as_float(best_ask or best_bid, 0.0)
        if price <= 0:
            raise ValueError("KuCoin REST ticker returned invalid price")
        if best_bid <= 0:
            best_bid = price
        if best_ask <= 0:
            best_ask = price
        return {
            "price": price,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "size": _as_float(data.get("size"), 0.0),
            "time": _as_float(data.get("time"), float(int(time.time() * 1000))),
            "source": "kucoin_rest",
        }

    def _fetch_orderbook_via_kucoin_rest(self, symbol: str, depth: int) -> Dict[str, Any]:
        level = 20 if int(depth) <= 20 else 100
        payload = self._public_get_json(
            f"{settings.kucoin.base_url}/api/v1/market/orderbook/level2_{level}",
            params={"symbol": symbol},
            timeout_sec=self._http_timeout_seconds,
        )
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        bids_raw = data.get("bids", []) if isinstance(data, dict) else []
        asks_raw = data.get("asks", []) if isinstance(data, dict) else []
        bids = [(float(b[0]), float(b[1])) for b in bids_raw[:depth] if len(b) >= 2]
        asks = [(float(a[0]), float(a[1])) for a in asks_raw[:depth] if len(a) >= 2]
        if not bids or not asks:
            raise ValueError("KuCoin REST orderbook returned no levels")
        return {
            "bids": bids,
            "asks": asks,
            "time": _as_float(data.get("time"), float(int(time.time() * 1000))),
            "source": "authenticated_orderbook",
        }

    def _orderbook_from_ticker(self, symbol: str) -> Dict[str, Any]:
        ticker = self.get_ticker(symbol)
        bid = _as_float(ticker.get("best_bid"), _as_float(ticker.get("price"), 0.0))
        ask = _as_float(ticker.get("best_ask"), _as_float(ticker.get("price"), 0.0))
        size = max(0.0, _as_float(ticker.get("size"), 0.0))
        if bid <= 0 or ask <= 0:
            raise ValueError("Ticker fallback missing best bid/ask")
        return {
            "bids": [(bid, size)],
            "asks": [(ask, size)],
            "time": _as_float(ticker.get("time"), float(int(time.time() * 1000))),
            "source": "public_ticker",
        }

    @staticmethod
    def _derive_market_data_source(ticker_source: str, orderbook_source: str) -> str:
        ticker_src = str(ticker_source or "").lower()
        book_src = str(orderbook_source or "").lower()
        if ticker_src == "market_data_service":
            return "market_data_service"
        if book_src in {"authenticated_orderbook"}:
            return "authenticated_orderbook"
        if "kucoin" in book_src and "orderbook" in book_src:
            return "authenticated_orderbook"
        if KuCoinDataFetcher._is_live_ticker_source(ticker_src):
            return "market_data_service"
        if ticker_src and ticker_src != "startup":
            return "unavailable"
        return "market_data_service"

    def _refresh_status(
        self,
        symbol: str,
        *,
        ticker_source: Optional[str] = None,
        orderbook_source: Optional[str] = None,
        market_data_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        key = str(symbol or ACTIVE_SYMBOL).upper()
        existing = self._status_by_symbol.get(
            key,
            {
                "symbol": key,
                "ticker_source": "unknown",
                "orderbook_source": "unknown",
                "market_data_source": "market_data_service",
            },
        )
        if ticker_source is not None:
            existing["ticker_source"] = str(ticker_source)
        if orderbook_source is not None:
            existing["orderbook_source"] = str(orderbook_source)
        if market_data_source is None:
            market_data_source = self._derive_market_data_source(
                str(existing.get("ticker_source", "")),
                str(existing.get("orderbook_source", "")),
            )
        existing["market_data_source"] = str(market_data_source)
        existing["synthetic_active"] = False
        existing["allow_synthetic_orderbook"] = False
        existing["allow_synthetic_ticker"] = False
        existing["allow_synthetic_klines"] = False
        existing["credentials_present"] = bool(self._credentials_present)
        existing["auth_required"] = bool(self._auth_required)
        existing["auth_valid"] = self._auth_valid
        existing["auth_error"] = self._auth_error
        existing["public_only_mode"] = bool(self._public_only_mode or self._offline_mode)
        existing["offline_mode"] = bool(self._offline_mode)
        health = self._compute_market_data_health(key, existing)
        existing.update(health)
        existing["can_trade"] = bool(existing.get("trading_enabled", False))
        existing["updated_at"] = datetime.utcnow().isoformat()
        self._status_by_symbol[key] = existing
        return dict(existing)

    def _run_startup_auth_check(self, force: bool = False) -> Dict[str, Any]:
        if self._startup_checked and not force:
            return self.get_market_data_status(ACTIVE_SYMBOL)

        self._startup_checked = True
        requires_auth = bool(self._auth_required)
        auth_check_enabled = bool(getattr(self._config, "auth_check_on_startup", True))

        if not self._credentials_present:
            self._auth_valid = False if requires_auth else None
            self._auth_error = (
                "KuCoin credentials missing. Configure KUCOIN_API_KEY/KUCOIN_API_SECRET/"
                "KUCOIN_API_PASSPHRASE (or KUCOIN_KEY/KUCOIN_SECRET/KUCOIN_PASSPHRASE, "
                "or KC_API_KEY/KC_API_SECRET/KC_API_PASSPHRASE)."
            )
            self._public_only_mode = True
            self._refresh_status(ACTIVE_SYMBOL)
            return self.get_market_data_status(ACTIVE_SYMBOL)

        if not auth_check_enabled or self.user_client is None:
            self._auth_valid = None
            self._auth_error = ""
            self._public_only_mode = self.user_client is None
            self._refresh_status(ACTIVE_SYMBOL)
            return self.get_market_data_status(ACTIVE_SYMBOL)

        try:
            _ = self.user_client.get_account_list(currency="USDT", account_type="trade")
            self._auth_valid = True
            self._auth_error = ""
            self._public_only_mode = False
        except Exception as exc:
            self._auth_valid = False
            self._auth_error = str(exc)
            self._public_only_mode = True
            self.trade_client = None
            self.user_client = None
            logger.error(
                "KuCoin authenticated startup check failed. Falling back to public-only market data: {}",
                exc,
            )

        self._refresh_status(ACTIVE_SYMBOL)
        return self.get_market_data_status(ACTIVE_SYMBOL)

    def startup_diagnostics(self, trading_mode: str = "PAPER") -> Dict[str, Any]:
        mode = str(trading_mode or "PAPER").upper()
        requires_auth = bool(self._auth_required or mode == "LIVE")
        self._auth_required = requires_auth
        status = self._run_startup_auth_check(force=True)
        market_probe_errors: List[str] = []
        try:
            self.get_ticker(ACTIVE_SYMBOL)
        except Exception as exc:
            market_probe_errors.append(f"ticker_probe_failed:{exc}")
        try:
            self.get_orderbook(ACTIVE_SYMBOL, depth=20)
        except Exception as exc:
            market_probe_errors.append(f"orderbook_probe_failed:{exc}")
        status = self.get_market_data_status(ACTIVE_SYMBOL)
        status["trading_mode"] = mode
        status["auth_required"] = requires_auth
        status["credentials_present"] = bool(self._credentials_present)
        status["can_trade"] = bool(status.get("trading_enabled", False))
        if requires_auth and status.get("auth_valid") is not True:
            status["startup_error"] = (
                status.get("auth_error")
                or "KuCoin authenticated trading is required but credentials are invalid."
            )
        elif not bool(status.get("market_data_ready", False)):
            status["startup_error"] = str(
                status.get("market_data_warning")
                or "Market data is not ready for trading."
            )
            if market_probe_errors:
                status["startup_error"] = f"{status['startup_error']} ({' | '.join(market_probe_errors)})"
        else:
            status["startup_error"] = ""
        return status

    def get_market_data_status(self, symbol: str = ACTIVE_SYMBOL) -> Dict[str, Any]:
        key = str(symbol or ACTIVE_SYMBOL).upper()
        status = self._refresh_status(key)
        return dict(status)

    def is_market_data_ready(
        self,
        symbol: str = ACTIVE_SYMBOL,
        *,
        max_age_seconds: Optional[float] = None,
    ) -> Tuple[bool, str]:
        status = self.get_market_data_status(symbol)
        if not bool(status.get("market_data_ready", False)):
            return False, str(status.get("market_data_warning") or "market_data_not_ready")

        if max_age_seconds is None:
            return True, "ready"

        try:
            max_age = max(1.0, float(max_age_seconds))
        except Exception:
            max_age = float(self._market_data_max_age_seconds)
        age = status.get("data_age_seconds")
        if age is None:
            return False, "market_data_age_unknown"
        if _as_float(age, 1e12) > max_age:
            return False, f"stale_market_data:{_as_float(age, 0.0):.1f}s>{max_age:.1f}s"
        return True, "ready"

    def _fetch_ticker_via_binance_rest(self, symbol: str) -> Dict[str, Any]:
        binance_symbol = self._to_binance_symbol(symbol)
        book_payload = self._public_get_json(
            "https://api.binance.com/api/v3/ticker/bookTicker",
            params={"symbol": binance_symbol},
            timeout_sec=self._http_timeout_seconds,
        )
        price_payload = self._public_get_json(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": binance_symbol},
            timeout_sec=self._http_timeout_seconds,
        )
        price = _as_float(price_payload.get("price"), 0.0)
        best_bid = _as_float(book_payload.get("bidPrice"), price)
        best_ask = _as_float(book_payload.get("askPrice"), price)
        if price <= 0:
            price = _as_float(best_ask or best_bid, 0.0)
        if price <= 0:
            raise ValueError("Binance REST ticker returned invalid price")
        if best_bid <= 0:
            best_bid = price
        if best_ask <= 0:
            best_ask = price
        return {
            "price": price,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "size": 0.0,
            "time": float(int(time.time() * 1000)),
            "source": "binance_rest",
        }

    def _fetch_klines_via_kucoin_rest(
        self,
        symbol: str,
        timeframe: str,
        start_dt: Optional[datetime],
        end_dt: Optional[datetime],
    ) -> pd.DataFrame:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "type": TIMEFRAME_MAP.get(timeframe, timeframe),
        }
        if start_dt:
            params["startAt"] = int(start_dt.timestamp())
        if end_dt:
            params["endAt"] = int(end_dt.timestamp())

        payload = self._public_get_json(
            f"{settings.kucoin.base_url}/api/v1/market/candles",
            params=params,
            timeout_sec=6.0,
        )
        raw = payload.get("data", []) if isinstance(payload, dict) else []
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"],
        )
        for col in ["open", "close", "high", "low", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df.dropna(subset=["timestamp", "open", "close", "high", "low", "volume"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        if not df.empty:
            self._last_prices[str(symbol or ACTIVE_SYMBOL).upper()] = float(df["close"].iloc[-1])
        return df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_top_usdt_pairs(self, top_n: int = 30) -> List[Dict[str, Any]]:
        """Get top N USDT trading pairs by 24h volume."""
        n = max(1, int(top_n))
        if self._offline_mode or self.market is None or self._public_only_mode:
            pairs = self._fallback_pairs(n)
            logger.info(
                "Using offline fallback universe: {}",
                [p.get("symbol") for p in pairs],
            )
            return pairs

        try:
            tickers = self.market.get_all_tickers()
            payload = tickers.get("ticker", []) if isinstance(tickers, dict) else []
            ranked: List[Dict[str, Any]] = []
            for row in payload:
                symbol = str(row.get("symbol", "")).upper()
                if not symbol.endswith("-USDT"):
                    continue
                vol_value = _as_float(
                    row.get("volValue", row.get("vol", 0.0)),
                    0.0,
                )
                ranked.append({"symbol": symbol, "volValue": str(vol_value)})
            ranked.sort(key=lambda item: _as_float(item.get("volValue"), 0.0), reverse=True)
            if not ranked:
                return self._fallback_pairs(n)
            return ranked[:n]
        except Exception as exc:
            self._maybe_switch_public_mode_from_error(exc=exc)
            logger.warning("Failed to fetch top USDT pairs from KuCoin, using fallback: {}", exc)
            return self._fallback_pairs(n)

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get exchange filters: tick size, lot size, min order."""
        self._assert_xrp_symbol(symbol)
        if self._offline_mode or self.market is None or self._public_only_mode:
            return self._fallback_symbol_info(symbol)

        try:
            getter = getattr(self.market, "get_symbol", None)
            if callable(getter):
                s = getter(symbol) or {}
                if s.get("symbol") == symbol:
                    return {
                        "symbol": s["symbol"],
                        "base_currency": s["baseCurrency"],
                        "quote_currency": s["quoteCurrency"],
                        "base_min_size": _as_float(s.get("baseMinSize", 0.0)),
                        "base_max_size": _as_float(s.get("baseMaxSize", 0.0)),
                        "base_increment": _as_float(s.get("baseIncrement", 0.0)),
                        "price_increment": _as_float(s.get("priceIncrement", 0.0)),
                        "quote_min_size": _as_float(s.get("quoteMinSize", 0.0)),
                        "quote_increment": _as_float(s.get("quoteIncrement", 0.0)),
                        "fee_currency": s.get("feeCurrency", "USDT"),
                    }
        except Exception as exc:
            self._maybe_switch_public_mode_from_error(exc=exc)
            logger.warning(f"Symbol info fallback for {symbol}: {exc}")

        return self._fallback_symbol_info(symbol)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def fetch_klines(
        self,
        symbol: str,
        timeframe: str = "1hour",
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV klines for a symbol."""
        if self._offline_mode or self.market is None or self._public_only_mode:
            try:
                return self._fetch_klines_via_kucoin_rest(symbol, timeframe, start_dt, end_dt)
            except Exception as exc:
                logger.error("REST klines failed for {} {}: {}", symbol, timeframe, exc)
                raise RuntimeError("Live market data unavailable") from exc

        kline_type = TIMEFRAME_MAP.get(timeframe, timeframe)
        params: Dict[str, Any] = {"symbol": symbol, "kline_type": kline_type}
        if start_dt:
            params["startAt"] = int(start_dt.timestamp())
        if end_dt:
            params["endAt"] = int(end_dt.timestamp())

        try:
            raw = self.market.get_kline(**params)
            if not raw:
                return pd.DataFrame()
            df = pd.DataFrame(
                raw,
                columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"],
            )
            for col in ["open", "close", "high", "low", "volume", "turnover"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
            df.dropna(subset=["timestamp", "open", "close", "high", "low", "volume"], inplace=True)
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            if not df.empty:
                self._last_prices[str(symbol or ACTIVE_SYMBOL).upper()] = float(df["close"].iloc[-1])
            return df
        except Exception as exc:
            self._maybe_switch_public_mode_from_error(exc=exc)
            logger.error("fetch_klines failed for {} {}: {}", symbol, timeframe, exc)
            raise RuntimeError("Live market data unavailable") from exc

    def fetch_history(
        self,
        symbol: str,
        timeframe: str = "1hour",
        months: int = 6,
    ) -> pd.DataFrame:
        """Fetch extended history by paginating through time windows."""
        self._assert_xrp_symbol(symbol)
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=max(1, int(months)) * 30)
        all_dfs: List[pd.DataFrame] = []
        interval_sec = TIMEFRAME_SECONDS.get(timeframe, 3600)
        chunk_duration = timedelta(seconds=interval_sec * 1400)
        current_start = start_dt

        while current_start < end_dt:
            current_end = min(current_start + chunk_duration, end_dt)
            df = self.fetch_klines(symbol, timeframe, current_start, current_end)
            if not df.empty:
                all_dfs.append(df)
            current_start = current_end
            time.sleep(0.05)

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        result.drop_duplicates(subset=["timestamp"], inplace=True)
        result.sort_values("timestamp", inplace=True)
        result.reset_index(drop=True, inplace=True)
        logger.info(f"Fetched {len(result)} candles for {symbol} {timeframe}")
        return result

    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Get current order book."""
        self._assert_xrp_symbol(symbol)
        _ = depth
        ticker = self.get_ticker(symbol)
        bid = _as_float(ticker.get("best_bid"), _as_float(ticker.get("price"), 0.0))
        ask = _as_float(ticker.get("best_ask"), _as_float(ticker.get("price"), 0.0))
        size = max(0.0, _as_float(ticker.get("size"), 0.0))
        if bid <= 0 or ask <= 0:
            self._refresh_status(symbol, orderbook_source="unavailable")
            raise RuntimeError("Live market data unavailable")
        orderbook = {
            "bids": [(bid, size)],
            "asks": [(ask, size)],
            "time": _as_float(ticker.get("time"), float(int(time.time() * 1000))),
            "source": "market_data_service",
        }
        self._refresh_status(symbol, orderbook_source="market_data_service")
        return orderbook

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data."""
        self._assert_xrp_symbol(symbol)
        symbol_key = str(symbol).upper()
        if self._is_market_halted(symbol_key):
            self._refresh_status(symbol_key, ticker_source="unavailable")
            raise RuntimeError("Live market data unavailable")

        ticker = self._resolve_live_ticker(symbol_key)
        if ticker is None:
            self._refresh_status(symbol_key, ticker_source="unavailable")
            raise RuntimeError("Live market data unavailable")

        ticker.setdefault("symbol", symbol_key)
        ticker.setdefault("exchange", self._market_exchange)
        ticker.setdefault("source", "market_data_service")
        self._ingest_market_ticker(ticker)

        self._refresh_status(
            symbol_key,
            ticker_source="market_data_service",
            market_data_source="market_data_service",
        )
        return ticker

    def get_spread(self, symbol: str) -> Tuple[float, float]:
        """Get bid-ask spread in absolute and percentage terms."""
        self._assert_xrp_symbol(symbol)
        ticker = self.get_ticker(symbol)
        bid = float(ticker["best_bid"])
        ask = float(ticker["best_ask"])
        spread_abs = max(0.0, ask - bid)
        spread_pct = spread_abs / ask if ask > 0 else 0.0
        return spread_abs, spread_pct

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def get_balance(self, currency: str = "USDT") -> float:
        """Get available balance for a currency."""
        if self._offline_mode or self.user_client is None:
            if currency.upper() == "USDT":
                return float(os.getenv("PAPER_BALANCE", settings.trading.paper_starting_balance))
            return 0.0

        accounts = self.user_client.get_account_list(currency=currency, account_type="trade")
        for acc in accounts:
            if acc["currency"] == currency:
                return _as_float(acc.get("available", 0.0), 0.0)
        return 0.0

    def get_all_balances(self) -> Dict[str, float]:
        """Get all non-zero trade balances."""
        if self._offline_mode or self.user_client is None:
            return {"USDT": float(os.getenv("PAPER_BALANCE", settings.trading.paper_starting_balance))}

        accounts = self.user_client.get_account_list(account_type="trade")
        return {
            acc["currency"]: _as_float(acc.get("available", 0.0), 0.0)
            for acc in accounts
            if _as_float(acc.get("available", 0.0), 0.0) > 0.0
        }
