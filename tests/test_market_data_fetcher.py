from __future__ import annotations

import pytest

from data.fetcher import KuCoinDataFetcher


def test_orderbook_falls_back_to_public_ticker_when_synthetic_disabled() -> None:
    fetcher = KuCoinDataFetcher()
    fetcher._orderbook_retry_attempts = 1
    fetcher._allow_synthetic_orderbook = False

    def _boom(*_args, **_kwargs):
        raise RuntimeError("orderbook api down")

    fetcher._fetch_orderbook_via_kucoin_rest = _boom  # type: ignore[method-assign]
    fetcher.get_ticker = lambda _symbol: {  # type: ignore[method-assign]
        "price": 1.0,
        "best_bid": 0.99,
        "best_ask": 1.01,
        "size": 10.0,
        "time": 1.0,
        "source": "kucoin_rest",
    }

    book = fetcher.get_orderbook("XRP-USDT", depth=20)
    assert book["source"] == "public_ticker"
    status = fetcher.get_market_data_status("XRP-USDT")
    assert status["market_data_source"] == "public_ticker"
    assert status["synthetic_active"] is False


def test_orderbook_raises_when_all_fallbacks_fail_and_synthetic_disabled() -> None:
    fetcher = KuCoinDataFetcher()
    fetcher._orderbook_retry_attempts = 1
    fetcher._allow_synthetic_orderbook = False

    def _boom(*_args, **_kwargs):
        raise RuntimeError("down")

    fetcher._fetch_orderbook_via_kucoin_rest = _boom  # type: ignore[method-assign]
    fetcher._orderbook_from_ticker = _boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="ALLOW_SYNTHETIC_ORDERBOOK=false"):
        fetcher.get_orderbook("XRP-USDT", depth=20)


def test_orderbook_can_use_synthetic_when_explicitly_enabled() -> None:
    fetcher = KuCoinDataFetcher()
    fetcher._orderbook_retry_attempts = 1
    fetcher._allow_synthetic_orderbook = True

    def _boom(*_args, **_kwargs):
        raise RuntimeError("down")

    fetcher._fetch_orderbook_via_kucoin_rest = _boom  # type: ignore[method-assign]
    fetcher._orderbook_from_ticker = _boom  # type: ignore[method-assign]

    book = fetcher.get_orderbook("XRP-USDT", depth=5)
    assert book["source"] == "synthetic"
    status = fetcher.get_market_data_status("XRP-USDT")
    assert status["market_data_source"] == "synthetic"
    assert status["synthetic_active"] is True


def test_startup_diagnostics_blocks_trading_when_auth_required_and_missing_credentials() -> None:
    fetcher = KuCoinDataFetcher()
    fetcher._credentials_present = False
    fetcher._auth_required = True

    status = fetcher.startup_diagnostics("PAPER")
    assert status["auth_required"] is True
    assert status["can_trade"] is False
    assert bool(status.get("startup_error"))


def test_ticker_switches_to_public_only_after_auth_failure() -> None:
    fetcher = KuCoinDataFetcher()
    fetcher._offline_mode = False
    fetcher._public_only_mode = False
    fetcher._credentials_present = True

    class _AuthFailMarket:
        @staticmethod
        def get_ticker(_symbol: str):
            raise RuntimeError('401-{"code":"400003","msg":"KC-API-KEY not exists"}')

    fetcher.market = _AuthFailMarket()
    fetcher._fetch_ticker_via_kucoin_rest = lambda _symbol: {  # type: ignore[method-assign]
        "price": 1.0,
        "best_bid": 0.99,
        "best_ask": 1.01,
        "size": 10.0,
        "time": 1.0,
        "source": "kucoin_rest",
    }

    ticker = fetcher.get_ticker("XRP-USDT")
    assert ticker["source"] == "kucoin_rest"
    assert fetcher._public_only_mode is True
    status = fetcher.get_market_data_status("XRP-USDT")
    assert status["public_only_mode"] is True


def test_ticker_uses_last_known_price_when_network_sources_fail() -> None:
    fetcher = KuCoinDataFetcher()
    fetcher._offline_mode = True
    fetcher._public_only_mode = True
    fetcher._last_prices["XRP-USDT"] = 1.2345

    def _boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    fetcher._fetch_ticker_via_kucoin_rest = _boom  # type: ignore[method-assign]
    fetcher._fetch_ticker_via_binance_rest = _boom  # type: ignore[method-assign]

    ticker = fetcher.get_ticker("XRP-USDT")
    assert ticker["source"] == "last_known"
    assert ticker["price"] == pytest.approx(1.2345)
