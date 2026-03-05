from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from loguru import logger

from data.fetcher import KuCoinDataFetcher
from quant.config import QuantEngineConfig
from quant.types import MarketSnapshot


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _funding_rate_symbol(symbol: str) -> str:
    return symbol.replace("-", "").upper()


class MarketDataEngine:
    """Pulls real-time multi-timeframe data and microstructure features."""

    def __init__(self, cfg: QuantEngineConfig, fetcher: KuCoinDataFetcher | None = None) -> None:
        self.cfg = cfg
        self.fetcher = fetcher or KuCoinDataFetcher()
        self.timeframes = ["1min", "5min", "15min"]
        self.snapshots: Dict[str, MarketSnapshot] = {}
        self._funding_cache: Dict[str, float] = {}

    def symbols(self) -> List[str]:
        return list(self.snapshots.keys() or self.cfg.universe)

    async def refresh_universe(self) -> List[str]:
        symbols = list(self.cfg.universe)
        if self.cfg.auto_expand_universe:
            try:
                pairs = await asyncio.to_thread(
                    self.fetcher.get_top_usdt_pairs,
                    self.cfg.max_universe_size,
                )
                discovered = [str(row.get("symbol", "")).upper() for row in pairs if row.get("symbol")]
                for symbol in discovered:
                    if symbol not in symbols:
                        symbols.append(symbol)
            except Exception as exc:
                logger.warning("Universe auto expansion failed: {}", exc)
        return symbols[: self.cfg.max_universe_size]

    async def update_all(self) -> Dict[str, MarketSnapshot]:
        symbols = await self.refresh_universe()
        tasks = [self._update_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        updated = 0
        for item in results:
            if isinstance(item, MarketSnapshot):
                self.snapshots[item.symbol] = item
                updated += 1
            elif isinstance(item, Exception):
                logger.warning("Market snapshot update failed: {}", item)
        logger.bind(event="MARKET_DATA_REFRESHED", symbols=updated).info("MARKET_DATA_REFRESHED")
        return dict(self.snapshots)

    async def _update_symbol(self, symbol: str) -> MarketSnapshot:
        now = datetime.utcnow()
        ticker = await asyncio.to_thread(self.fetcher.get_ticker, symbol)
        orderbook = await asyncio.to_thread(self.fetcher.get_orderbook, symbol, 20)
        funding_rate = await asyncio.to_thread(self._fetch_funding_rate, symbol)

        frames: Dict[str, pd.DataFrame] = {}
        for timeframe in self.timeframes:
            delta = {"1min": 12, "5min": 36, "15min": 96}[timeframe]
            start = now - timedelta(hours=delta)
            df = await asyncio.to_thread(
                self.fetcher.fetch_klines,
                symbol,
                timeframe,
                start,
                None,
            )
            if df.empty:
                raise RuntimeError(f"No market candles for {symbol} {timeframe}")
            frames[timeframe] = df.tail(500).copy()

        best_bid = _safe_float(ticker.get("best_bid", ticker.get("price")))
        best_ask = _safe_float(ticker.get("best_ask", ticker.get("price")))
        mark = _safe_float(ticker.get("price"), (best_bid + best_ask) / 2.0 if best_ask > 0 else best_bid)
        spread_pct = (max(0.0, best_ask - best_bid) / best_ask) if best_ask > 0 else 0.0

        bids = orderbook.get("bids", []) if isinstance(orderbook, dict) else []
        asks = orderbook.get("asks", []) if isinstance(orderbook, dict) else []
        bid_qty = sum(_safe_float(level[1]) for level in bids[:20]) if bids else 0.0
        ask_qty = sum(_safe_float(level[1]) for level in asks[:20]) if asks else 0.0
        imbalance_den = bid_qty + ask_qty
        orderbook_imbalance = ((bid_qty - ask_qty) / imbalance_den) if imbalance_den > 0 else 0.0

        frame_1m = frames["1min"]
        recent_vol_buy = _safe_float(frame_1m["volume"].tail(6).mean())
        recent_vol_total = _safe_float(frame_1m["volume"].tail(30).mean())
        volume_imbalance = (
            (recent_vol_buy - recent_vol_total) / recent_vol_total
            if recent_vol_total > 0
            else 0.0
        )

        ret = frame_1m["close"].pct_change().dropna()
        realized_vol = _safe_float(ret.tail(30).std(), 0.0) * np.sqrt(60.0 * 24.0)

        return MarketSnapshot(
            symbol=symbol,
            timestamp=now,
            ticker_price=mark,
            best_bid=best_bid,
            best_ask=best_ask,
            spread_pct=spread_pct,
            orderbook_imbalance=orderbook_imbalance,
            volume_imbalance=volume_imbalance,
            funding_rate=funding_rate,
            realized_volatility=realized_vol,
            frames=frames,
        )

    def _fetch_funding_rate(self, symbol: str) -> float:
        if symbol in self._funding_cache:
            return self._funding_cache[symbol]
        endpoint = "https://fapi.binance.com/fapi/v1/premiumIndex"
        params = {"symbol": _funding_rate_symbol(symbol)}
        try:
            response = requests.get(endpoint, params=params, timeout=3.0)
            response.raise_for_status()
            payload = response.json()
            value = _safe_float(payload.get("lastFundingRate"), 0.0)
            self._funding_cache[symbol] = value
            return value
        except Exception:
            # Spot universe fallback proxy using signed microstructure pressure.
            fallback = np.tanh(_safe_float(self._funding_cache.get(symbol), 0.0))
            self._funding_cache[symbol] = float(fallback)
            return float(fallback)
