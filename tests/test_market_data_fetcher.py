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
