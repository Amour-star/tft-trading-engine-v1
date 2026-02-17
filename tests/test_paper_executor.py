"""Tests for paper trading execution and mode safety guards."""
from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import TradingConfig
from execution.paper_executor import PaperExecutor


class MockFetcher:
    def __init__(self) -> None:
        self.prices = {
            "BTC-USDT": 100.0,
            "ETH-USDT": 200.0,
        }

    def get_ticker(self, symbol: str):
        price = self.prices[symbol]
        return {
            "price": price,
            "best_ask": price,
            "best_bid": price,
        }


class _ForbiddenClient:
    def __getattr__(self, name):
        raise AssertionError(f"Exchange order API should not be called in PAPER mode: {name}")


class GuardedFetcher(MockFetcher):
    def __init__(self) -> None:
        super().__init__()
        self.trade_client = _ForbiddenClient()
        self.user_client = _ForbiddenClient()


def test_paper_buy_sell_lifecycle(tmp_path: Path):
    fetcher = MockFetcher()
    db_path = tmp_path / "paper_lifecycle.db"
    executor = PaperExecutor(
        fetcher,
        starting_balance=1000.0,
        fee_rate=0.001,
        slippage_bps=0.0,
        db_path=str(db_path),
    )

    buy = executor.buy("BTC-USDT", qty=1.0, price=100.0)
    assert buy["status"] == "filled"
    assert buy["fee"] == pytest.approx(0.1)
    assert executor.get_balance() == pytest.approx(899.9)

    positions = executor.get_positions()
    assert positions["BTC-USDT"]["quantity"] == pytest.approx(1.0)
    assert positions["BTC-USDT"]["avg_entry_price"] == pytest.approx(100.0)

    sell = executor.sell("BTC-USDT", qty=1.0, price=110.0)
    assert sell["fee"] == pytest.approx(0.11)
    assert sell["realized_pnl"] == pytest.approx(9.89)
    assert executor.get_balance() == pytest.approx(1009.79)
    assert executor.get_realized_pnl() == pytest.approx(9.89)
    assert executor.get_positions() == {}
    assert len(executor.trade_history) == 2


def test_paper_fee_calculation(tmp_path: Path):
    fetcher = MockFetcher()
    executor = PaperExecutor(
        fetcher,
        starting_balance=500.0,
        fee_rate=0.002,
        slippage_bps=0.0,
        db_path=str(tmp_path / "paper_fee.db"),
    )

    buy = executor.buy("ETH-USDT", qty=1.0, price=200.0)
    assert buy["fee"] == pytest.approx(0.4)
    assert executor.get_balance() == pytest.approx(299.6)


def test_paper_pnl_close_position(tmp_path: Path):
    fetcher = MockFetcher()
    executor = PaperExecutor(
        fetcher,
        starting_balance=1000.0,
        fee_rate=0.001,
        slippage_bps=0.0,
        db_path=str(tmp_path / "paper_pnl.db"),
    )

    executor.buy("BTC-USDT", qty=1.0, price=100.0)
    close = executor.close_position("BTC-USDT")

    # No explicit price passed, so it uses current mock market (100) with sell fee only.
    assert close["realized_pnl"] == pytest.approx(-0.1)
    assert executor.get_positions() == {}


def test_paper_persistence_reload(tmp_path: Path):
    fetcher = MockFetcher()
    db_path = tmp_path / "paper_reload.db"

    first = PaperExecutor(
        fetcher,
        starting_balance=1000.0,
        fee_rate=0.001,
        slippage_bps=0.0,
        db_path=str(db_path),
    )
    first.buy("BTC-USDT", qty=0.5, price=100.0)

    second = PaperExecutor(
        fetcher,
        starting_balance=5000.0,
        fee_rate=0.001,
        slippage_bps=0.0,
        db_path=str(db_path),
    )
    positions = second.get_positions()
    assert "BTC-USDT" in positions
    assert positions["BTC-USDT"]["quantity"] == pytest.approx(0.5)
    assert second.get_balance() == pytest.approx(949.95)


def test_paper_mode_never_uses_exchange_order_api(tmp_path: Path):
    fetcher = GuardedFetcher()
    executor = PaperExecutor(
        fetcher,
        starting_balance=1000.0,
        fee_rate=0.001,
        slippage_bps=0.0,
        db_path=str(tmp_path / "paper_guard.db"),
    )

    executor.buy("BTC-USDT", qty=0.1, price=100.0)
    executor.sell("BTC-USDT", qty=0.1, price=105.0)


def test_trading_mode_validation_unknown():
    with pytest.raises(ValueError):
        TradingConfig(trading_mode="UNKNOWN")


def test_trading_mode_validation_live_guard():
    with pytest.raises(ValueError):
        TradingConfig(trading_mode="LIVE", allow_live_trading="NO")

    cfg = TradingConfig(trading_mode="LIVE", allow_live_trading="YES_I_UNDERSTAND")
    assert cfg.trading_mode == "LIVE"
    assert not cfg.paper_trading
