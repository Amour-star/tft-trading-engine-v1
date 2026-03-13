from __future__ import annotations

import time

import pytest

from data.fetcher import KuCoinDataFetcher


def _stub_market_bus(self) -> None:  # type: ignore[no-untyped-def]
    self._redis = None
    self._pubsub = None
    self._redis_connected = True
    self._redis_error = ""


def test_get_ticker_raises_when_live_market_data_missing(monkeypatch) -> None:
    monkeypatch.setattr(KuCoinDataFetcher, "_init_market_data_bus", _stub_market_bus)
    fetcher = KuCoinDataFetcher()
    fetcher._resolve_live_ticker = lambda _symbol: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Live market data unavailable"):
        fetcher.get_ticker.__wrapped__(fetcher, "XRP-USDT")


def test_get_ticker_reads_from_market_data_service(monkeypatch) -> None:
    monkeypatch.setattr(KuCoinDataFetcher, "_init_market_data_bus", _stub_market_bus)
    fetcher = KuCoinDataFetcher()

    now_ms = int(time.time() * 1000)
    fetcher._resolve_live_ticker = lambda _symbol: {  # type: ignore[method-assign]
        "symbol": "XRP-USDT",
        "exchange": "kucoin",
        "source": "market_data_service",
        "price": 1.0,
        "best_bid": 0.999,
        "best_ask": 1.001,
        "size": 100.0,
        "time": now_ms,
        "timestamp": now_ms,
        "volume": 100.0,
    }

    ticker = fetcher.get_ticker.__wrapped__(fetcher, "XRP-USDT")
    assert ticker["source"] == "market_data_service"
    assert ticker["price"] == pytest.approx(1.0)

    status = fetcher.get_market_data_status("XRP-USDT")
    assert status["market_data_source"] == "market_data_service"
    assert status["market_data_ready"] is True
    assert status["synthetic_active"] is False


def test_orderbook_is_derived_from_live_ticker(monkeypatch) -> None:
    monkeypatch.setattr(KuCoinDataFetcher, "_init_market_data_bus", _stub_market_bus)
    fetcher = KuCoinDataFetcher()
    fetcher.get_ticker = lambda _symbol: {  # type: ignore[method-assign]
        "price": 1.0,
        "best_bid": 0.99,
        "best_ask": 1.01,
        "size": 25.0,
        "time": 1.0,
        "source": "market_data_service",
    }

    book = fetcher.get_orderbook("XRP-USDT", depth=20)
    assert book["source"] == "market_data_service"
    assert book["bids"][0][0] == pytest.approx(0.99)
    assert book["asks"][0][0] == pytest.approx(1.01)


def test_market_data_ready_turns_false_when_price_is_stale(monkeypatch) -> None:
    monkeypatch.setattr(KuCoinDataFetcher, "_init_market_data_bus", _stub_market_bus)
    fetcher = KuCoinDataFetcher()
    fetcher._market_data_max_age_seconds = 5

    now = time.time()
    fetcher._last_valid_ticker["XRP-USDT"] = {
        "price": 1.0,
        "best_bid": 0.99,
        "best_ask": 1.01,
        "size": 10.0,
        "time": now * 1000,
        "source": "market_data_service",
        "updated_at": "2026-03-06T00:00:00",
        "received_at_unix": now - 30,
        "latency_ms": 0.0,
        "exchange": "kucoin",
    }
    fetcher._refresh_status(
        "XRP-USDT",
        ticker_source="market_data_service",
        market_data_source="market_data_service",
    )

    status = fetcher.get_market_data_status("XRP-USDT")
    assert status["market_data_ready"] is False
    assert "stale_market_data" in str(status.get("market_data_warning", ""))


def test_startup_diagnostics_reports_failure_when_live_feed_missing(monkeypatch) -> None:
    monkeypatch.setattr(KuCoinDataFetcher, "_init_market_data_bus", _stub_market_bus)
    fetcher = KuCoinDataFetcher()
    fetcher.get_ticker = lambda _symbol: (_ for _ in ()).throw(RuntimeError("Live market data unavailable"))  # type: ignore[method-assign]
    fetcher.get_orderbook = lambda _symbol, depth=20: (_ for _ in ()).throw(RuntimeError("Live market data unavailable"))  # type: ignore[method-assign]

    status = fetcher.startup_diagnostics("PAPER")
    assert status["can_trade"] is False
    assert "Live market data unavailable" in str(status.get("startup_error", ""))
