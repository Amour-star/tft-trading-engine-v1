"""
KuCoin market data fetcher.
Handles OHLCV retrieval, top pairs discovery, and real-time data fallbacks.
"""
from __future__ import annotations

import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
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
        self._allow_synthetic_orderbook = bool(getattr(cfg, "allow_synthetic_orderbook", False))
        self._orderbook_retry_attempts = max(1, int(getattr(cfg, "orderbook_retry_attempts", 3)))
        self._http_retry_attempts = max(1, int(getattr(cfg, "http_retry_attempts", 4)))
        self._http_timeout_seconds = max(1.0, float(getattr(cfg, "http_timeout_seconds", 4.0)))
        self._http_max_backoff_seconds = max(1.0, float(getattr(cfg, "http_max_backoff_seconds", 8.0)))

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
            ACTIVE_SYMBOL: 0.6,
        }

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

        if self._offline_mode:
            logger.warning("KuCoin fetcher running in offline-safe mode.")

        self._refresh_status(
            ACTIVE_SYMBOL,
            ticker_source="startup",
            orderbook_source="startup",
            market_data_source="public_ticker",
        )
        self._run_startup_auth_check(force=True)

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
        if symbol in self._last_prices:
            return self._last_prices[symbol]
        base = symbol.split("-")[0]
        seed = sum(ord(ch) for ch in base) % 1000
        price = max(0.05, float(seed))
        self._last_prices[symbol] = price
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
        self._last_prices[symbol] = float(df["close"].iloc[-1])
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
        price = _as_float(self._last_prices.get(symbol), 0.0)
        if price <= 0:
            return None
        spread = max(1e-8, price * 0.0005)
        return {
            "price": price,
            "best_bid": max(1e-8, price - spread),
            "best_ask": price + spread,
            "size": 0.0,
            "time": float(int(time.time() * 1000)),
            "source": "last_known",
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
            "source": "kucoin_rest_orderbook",
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
        if "synthetic" in ticker_src or "synthetic" in book_src:
            return "synthetic"
        if book_src in {"public_ticker", "ticker"}:
            return "public_ticker"
        if book_src:
            return "real"
        if ticker_src and ticker_src != "startup":
            return "public_ticker"
        return "public_ticker"

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
                "market_data_source": "public_ticker",
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
        existing["synthetic_active"] = str(existing["market_data_source"]).lower() == "synthetic"
        existing["allow_synthetic_orderbook"] = bool(self._allow_synthetic_orderbook)
        existing["credentials_present"] = bool(self._credentials_present)
        existing["auth_required"] = bool(self._auth_required)
        existing["auth_valid"] = self._auth_valid
        existing["auth_error"] = self._auth_error
        existing["public_only_mode"] = bool(self._public_only_mode or self._offline_mode)
        existing["offline_mode"] = bool(self._offline_mode)
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
        status["trading_mode"] = mode
        status["auth_required"] = requires_auth
        status["credentials_present"] = bool(self._credentials_present)
        status["can_trade"] = bool(not requires_auth or status.get("auth_valid") is True)
        if requires_auth and status.get("auth_valid") is not True:
            status["startup_error"] = (
                status.get("auth_error")
                or "KuCoin authenticated trading is required but credentials are invalid."
            )
        else:
            status["startup_error"] = ""
        return status

    def get_market_data_status(self, symbol: str = ACTIVE_SYMBOL) -> Dict[str, Any]:
        key = str(symbol or ACTIVE_SYMBOL).upper()
        status = self._status_by_symbol.get(key)
        if status is None:
            status = self._refresh_status(key)
        return dict(status)

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
            self._last_prices[symbol] = float(df["close"].iloc[-1])
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
                logger.warning(f"REST klines failed for {symbol} {timeframe}, using synthetic data: {exc}")
                return self._generate_synthetic_klines(symbol, timeframe, start_dt, end_dt)

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
                self._last_prices[symbol] = float(df["close"].iloc[-1])
            return df
        except Exception as exc:
            self._maybe_switch_public_mode_from_error(exc=exc)
            logger.warning(f"fetch_klines failed for {symbol} {timeframe}, using synthetic data: {exc}")
            return self._generate_synthetic_klines(symbol, timeframe, start_dt, end_dt)

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
        symbol_key = str(symbol).upper()

        failures: List[str] = []
        for attempt in range(1, self._orderbook_retry_attempts + 1):
            try:
                orderbook = self._fetch_orderbook_via_kucoin_rest(symbol, depth)
                self._refresh_status(
                    symbol_key,
                    orderbook_source=str(orderbook.get("source", "kucoin_rest_orderbook")),
                )
                return orderbook
            except Exception as exc:
                self._maybe_switch_public_mode_from_error(exc=exc)
                failures.append(f"attempt={attempt}: {exc}")
                if attempt < self._orderbook_retry_attempts:
                    backoff = min(0.5 * (2 ** (attempt - 1)), 5.0)
                    time.sleep(backoff)

        try:
            fallback = self._orderbook_from_ticker(symbol)
            self._refresh_status(symbol_key, orderbook_source="public_ticker")
            logger.warning(
                "Orderbook unavailable for {} after {} attempt(s); using public ticker fallback",
                symbol,
                self._orderbook_retry_attempts,
            )
            return fallback
        except Exception as fallback_exc:
            failures.append(f"public_ticker_fallback_failed: {fallback_exc}")

        if self._allow_synthetic_orderbook:
            synthetic = self._synthetic_orderbook(symbol, depth)
            self._refresh_status(symbol_key, orderbook_source="synthetic")
            logger.error(
                "Orderbook failed for {}. Synthetic fallback is enabled and will be used. errors={}",
                symbol,
                " | ".join(failures),
            )
            return synthetic

        self._refresh_status(symbol_key, orderbook_source="unavailable")
        raise RuntimeError(
            "Orderbook fetch failed and synthetic fallback is disabled "
            "(ALLOW_SYNTHETIC_ORDERBOOK=false). "
            f"symbol={symbol} errors={' | '.join(failures)}"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data."""
        self._assert_xrp_symbol(symbol)
        if self._offline_mode or self.market is None or self._public_only_mode:
            try:
                ticker = self._fetch_ticker_via_kucoin_rest(symbol)
                self._last_prices[symbol] = float(ticker["price"])
                self._refresh_status(symbol, ticker_source=str(ticker.get("source", "kucoin_rest")))
                return ticker
            except Exception as kucoin_exc:
                logger.warning(f"KuCoin REST ticker failed for {symbol}: {kucoin_exc}")
                try:
                    ticker = self._fetch_ticker_via_binance_rest(symbol)
                    self._last_prices[symbol] = float(ticker["price"])
                    self._refresh_status(symbol, ticker_source=str(ticker.get("source", "binance_rest")))
                    return ticker
                except Exception as binance_exc:
                    logger.warning(f"Binance REST ticker failed for {symbol}, falling back to cached/synthetic ticker: {binance_exc}")
                    cached = self._cached_ticker(symbol)
                    if cached is not None:
                        self._refresh_status(symbol, ticker_source="last_known")
                        return cached
                    ticker = self._synthetic_ticker(symbol)
                    self._refresh_status(symbol, ticker_source="synthetic")
                    return ticker

        try:
            t = self.market.get_ticker(symbol)
            price = _as_float(
                t.get("price")
                or t.get("last")
                or t.get("lastPrice")
                or t.get("lastTradedPrice")
                or 0.0,
                0.0,
            )
            if price <= 0:
                price = _as_float(
                    t.get("bestAsk")
                    or t.get("bestBid")
                    or t.get("bestAskPrice")
                    or t.get("bestBidPrice")
                    or t.get("sell")
                    or t.get("buy")
                    or 0.0,
                    self._base_price(symbol),
                )

            best_bid = _as_float(
                t.get("bestBid")
                or t.get("bestBidPrice")
                or t.get("buy")
                or t.get("best_bid")
                or price,
                price,
            )
            best_ask = _as_float(
                t.get("bestAsk")
                or t.get("bestAskPrice")
                or t.get("sell")
                or t.get("best_ask")
                or price,
                price,
            )
            if best_bid <= 0:
                best_bid = price
            if best_ask <= 0:
                best_ask = price

            self._last_prices[symbol] = price
            ticker = {
                "price": price,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "size": _as_float(t.get("size", 0.0), 0.0),
                "time": _as_float(t.get("time", int(time.time() * 1000)), int(time.time() * 1000)),
                "source": "kucoin_sdk",
            }
            self._refresh_status(symbol, ticker_source="kucoin_sdk")
            return ticker
        except Exception as exc:
            self._maybe_switch_public_mode_from_error(exc=exc)
            logger.warning(f"SDK ticker failed for {symbol}: {exc}")
            try:
                ticker = self._fetch_ticker_via_kucoin_rest(symbol)
                self._last_prices[symbol] = float(ticker["price"])
                self._refresh_status(symbol, ticker_source=str(ticker.get("source", "kucoin_rest")))
                return ticker
            except Exception as kucoin_exc:
                logger.warning(f"KuCoin REST ticker failed for {symbol}: {kucoin_exc}")
                try:
                    ticker = self._fetch_ticker_via_binance_rest(symbol)
                    self._last_prices[symbol] = float(ticker["price"])
                    self._refresh_status(symbol, ticker_source=str(ticker.get("source", "binance_rest")))
                    return ticker
                except Exception as binance_exc:
                    logger.warning(f"Binance REST ticker failed for {symbol}, falling back to cached/synthetic ticker: {binance_exc}")
                    cached = self._cached_ticker(symbol)
                    if cached is not None:
                        self._refresh_status(symbol, ticker_source="last_known")
                        return cached
                    ticker = self._synthetic_ticker(symbol)
                    self._refresh_status(symbol, ticker_source="synthetic")
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
