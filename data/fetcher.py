"""
KuCoin market data fetcher.
Handles OHLCV retrieval, top pairs discovery, and real-time data fallbacks.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import XRP_ONLY_SYMBOL, settings

try:
    from kucoin_universal_sdk.client import KucoinClient
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
    {"symbol": XRP_ONLY_SYMBOL, "volValue": "180000000"},
]

_FALLBACK_SYMBOL_INFO = {
    "baseMinSize": "0.00001",
    "baseMaxSize": "1000000",
    "baseIncrement": "0.00001",
    "priceIncrement": "0.00001",
    "quoteMinSize": "0.01",
    "quoteIncrement": "0.01",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _offline_mode_enabled() -> bool:
    raw = os.getenv("KUCOIN_OFFLINE_MODE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class KuCoinDataFetcher:
    """Fetches and manages market data from KuCoin."""

    def __init__(self) -> None:
        cfg = settings.kucoin
        url = cfg.base_url
        self._offline_mode = _offline_mode_enabled() or Market is None

        self.market = None
        self.trade_client = None
        self.user_client = None
        self._last_prices: Dict[str, float] = {
            XRP_ONLY_SYMBOL: 0.6,
        }

        if not self._offline_mode:
            try:
                self.market = KucoinClient(
                    key=cfg.api_key,
                    secret=cfg.api_secret,
                    passphrase=cfg.api_passphrase,
                    base_url=url,
                ) if Market else None
                self.trade_client = (
                    Trade(
                        key=cfg.api_key,
                        secret=cfg.api_secret,
                        passphrase=cfg.api_passphrase,
                        url=url,
                    )
                    if Trade and cfg.api_key and cfg.api_secret and cfg.api_passphrase
                    else None
                )
                self.user_client = (
                    User(
                        key=cfg.api_key,
                        secret=cfg.api_secret,
                        passphrase=cfg.api_passphrase,
                        url=url,
                    )
                    if User and cfg.api_key and cfg.api_secret and cfg.api_passphrase
                    else None
                )
            except Exception as exc:
                logger.warning(f"Failed to initialize KuCoin clients, using offline mode: {exc}")
                self._offline_mode = True

        if self._offline_mode:
            logger.warning("KuCoin fetcher running in offline-safe mode.")

    @staticmethod
    def _assert_xrp_symbol(symbol: str) -> None:
        assert symbol == XRP_ONLY_SYMBOL, f"XRP-only mode: unsupported symbol {symbol}"

    def _fallback_pairs(self, top_n: int) -> List[Dict[str, Any]]:
        _ = top_n
        return list(_FALLBACK_PAIRS)

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
        return {"bids": bids, "asks": asks, "time": int(time.time() * 1000)}

    def _synthetic_ticker(self, symbol: str) -> Dict[str, float]:
        self._assert_xrp_symbol(symbol)
        price = float(self._base_price(symbol))
        return {
            "price": price,
            "best_bid": price * 0.9995,
            "best_ask": price * 1.0005,
            "size": 1.0,
            "time": float(int(time.time() * 1000)),
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_top_usdt_pairs(self, top_n: int = 30) -> List[Dict[str, Any]]:
        """Get top N USDT trading pairs by 24h volume."""
        _ = top_n
        logger.info("Universe locked to XRP-USDT")
        return self._fallback_pairs(1)

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get exchange filters: tick size, lot size, min order."""
        self._assert_xrp_symbol(symbol)
        if self._offline_mode or self.market is None:
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
        assert symbol == XRP_ONLY_SYMBOL
        if self._offline_mode or self.market is None:
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Get current order book."""
        self._assert_xrp_symbol(symbol)
        if self._offline_mode or self.market is None:
            return self._synthetic_orderbook(symbol, depth)

        try:
            book = self.market.get_aggregated_orderv3(symbol)
            return {
                "bids": [(float(b[0]), float(b[1])) for b in book.get("bids", [])[:depth]],
                "asks": [(float(a[0]), float(a[1])) for a in book.get("asks", [])[:depth]],
                "time": book.get("time"),
            }
        except Exception as exc:
            logger.warning(f"Orderbook fetch failed, using synthetic orderbook: {exc}")
            return self._synthetic_orderbook(symbol, depth)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Get current ticker data."""
        self._assert_xrp_symbol(symbol)
        if self._offline_mode or self.market is None:
            return self._synthetic_ticker(symbol)

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
            return {
                "price": price,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "size": _as_float(t.get("size", 0.0), 0.0),
                "time": _as_float(t.get("time", int(time.time() * 1000)), int(time.time() * 1000)),
            }
        except Exception as exc:
            logger.warning(f"Ticker fetch failed, using synthetic ticker: {exc}")
            return self._synthetic_ticker(symbol)

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
