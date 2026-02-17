"""
KuCoin market data fetcher.
Handles OHLCV retrieval, top pairs discovery, and real-time streaming.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings

# KuCoin REST client
try:
    from kucoin.client import Market, Trade, User
except Exception as exc:
    Market = Trade = User = None  # type: ignore
    logger.warning(f"kucoin-python unavailable ({exc}). Install with: pip install kucoin-python")


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


class KuCoinDataFetcher:
    """Fetches and manages market data from KuCoin."""

    def __init__(self) -> None:
        cfg = settings.kucoin
        url = cfg.base_url

        self.market = Market(url=url) if Market else None
        self.trade_client = Trade(
            key=cfg.api_key,
            secret=cfg.api_secret,
            passphrase=cfg.api_passphrase,
            url=url,
        ) if Trade else None
        self.user_client = User(
            key=cfg.api_key,
            secret=cfg.api_secret,
            passphrase=cfg.api_passphrase,
            url=url,
        ) if User else None

    # ------------------------------------------------------------------
    # Top pairs discovery
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_top_usdt_pairs(self, top_n: int = 30) -> List[Dict[str, Any]]:
        """Get top N USDT trading pairs by 24h volume."""
        tickers = self.market.get_all_tickers()["ticker"]
        usdt_pairs = [
            t for t in tickers
            if t["symbol"].endswith("-USDT")
            and float(t.get("volValue", 0)) > settings.trading.min_volume_24h
        ]
        # Sort by volume descending
        usdt_pairs.sort(key=lambda x: float(x.get("volValue", 0)), reverse=True)
        return usdt_pairs[:top_n]

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get exchange filters: tick size, lot size, min order."""
        symbols = self.market.get_symbol_list()
        for s in symbols:
            if s["symbol"] == symbol:
                return {
                    "symbol": s["symbol"],
                    "base_currency": s["baseCurrency"],
                    "quote_currency": s["quoteCurrency"],
                    "base_min_size": float(s.get("baseMinSize", 0)),
                    "base_max_size": float(s.get("baseMaxSize", 0)),
                    "base_increment": float(s.get("baseIncrement", 0)),
                    "price_increment": float(s.get("priceIncrement", 0)),
                    "quote_min_size": float(s.get("quoteMinSize", 0)),
                    "quote_increment": float(s.get("quoteIncrement", 0)),
                    "fee_currency": s.get("feeCurrency", "USDT"),
                }
        raise ValueError(f"Symbol {symbol} not found on KuCoin")

    # ------------------------------------------------------------------
    # OHLCV data
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def fetch_klines(
        self,
        symbol: str,
        timeframe: str = "1hour",
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV klines for a symbol."""
        kline_type = TIMEFRAME_MAP.get(timeframe, timeframe)
        params: Dict[str, Any] = {"symbol": symbol, "kline_type": kline_type}
        if start_dt:
            params["startAt"] = int(start_dt.timestamp())
        if end_dt:
            params["endAt"] = int(end_dt.timestamp())

        raw = self.market.get_kline(**params)
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "close", "high", "low", "volume", "turnover"
        ])
        for col in ["open", "close", "high", "low", "volume", "turnover"]:
            df[col] = df[col].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def fetch_history(
        self,
        symbol: str,
        timeframe: str = "1hour",
        months: int = 6,
    ) -> pd.DataFrame:
        """Fetch extended history by paginating through time windows."""
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=months * 30)
        all_dfs: List[pd.DataFrame] = []
        interval_sec = TIMEFRAME_SECONDS.get(timeframe, 3600)
        # KuCoin returns max 1500 candles per request
        chunk_duration = timedelta(seconds=interval_sec * 1400)
        current_start = start_dt

        while current_start < end_dt:
            current_end = min(current_start + chunk_duration, end_dt)
            logger.debug(f"Fetching {symbol} {timeframe} from {current_start} to {current_end}")
            df = self.fetch_klines(symbol, timeframe, current_start, current_end)
            if not df.empty:
                all_dfs.append(df)
            current_start = current_end
            time.sleep(0.2)  # Rate limiting

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        result.drop_duplicates(subset=["timestamp"], inplace=True)
        result.sort_values("timestamp", inplace=True)
        result.reset_index(drop=True, inplace=True)
        logger.info(f"Fetched {len(result)} candles for {symbol} {timeframe}")
        return result

    # ------------------------------------------------------------------
    # Order book & ticker
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Get current order book."""
        book = self.market.get_aggregated_orderv3(symbol)
        return {
            "bids": [(float(b[0]), float(b[1])) for b in book.get("bids", [])[:depth]],
            "asks": [(float(a[0]), float(a[1])) for a in book.get("asks", [])[:depth]],
            "time": book.get("time"),
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Get current ticker data."""
        t = self.market.get_ticker(symbol)
        return {
            "price": float(t.get("price", 0)),
            "best_bid": float(t.get("bestBid", 0)),
            "best_ask": float(t.get("bestAsk", 0)),
            "size": float(t.get("size", 0)),
            "time": t.get("time"),
        }

    def get_spread(self, symbol: str) -> Tuple[float, float]:
        """Get bid-ask spread in absolute and percentage terms."""
        ticker = self.get_ticker(symbol)
        bid = ticker["best_bid"]
        ask = ticker["best_ask"]
        spread_abs = ask - bid
        spread_pct = spread_abs / ask if ask > 0 else 0
        return spread_abs, spread_pct

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def get_balance(self, currency: str = "USDT") -> float:
        """Get available balance for a currency."""
        accounts = self.user_client.get_account_list(currency=currency, account_type="trade")
        for acc in accounts:
            if acc["currency"] == currency:
                return float(acc.get("available", 0))
        return 0.0

    def get_all_balances(self) -> Dict[str, float]:
        """Get all non-zero trade balances."""
        accounts = self.user_client.get_account_list(account_type="trade")
        return {
            acc["currency"]: float(acc["available"])
            for acc in accounts
            if float(acc.get("available", 0)) > 0
        }
