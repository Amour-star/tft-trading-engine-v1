"""Tests for paper trading execution and mode safety guards."""
from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import TradingConfig
from execution.paper_executor import PaperExecutor
from engine.decision import TradeSignal


class MockFetcher:
    def __init__(self) -> None:
        self.prices = {
            "XRP-USDT": 100.0,
            "XRP-USDT": 200.0,
        }

    def get_ticker(self, symbol: str):
        price = self.prices[symbol]
        return {
            "price": price,
            "best_ask": price,
            "best_bid": price,
        }

    def get_symbol_info(self, symbol: str):
        base, quote = symbol.split("-")
        return {
            "symbol": symbol,
            "base_currency": base,
            "quote_currency": quote,
            "base_min_size": 0.00001,
            "base_max_size": 1_000_000.0,
            "base_increment": 0.00001,
            "price_increment": 0.00001,
            "quote_min_size": 0.01,
            "quote_increment": 0.01,
            "fee_currency": quote,
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

    buy = executor.buy("XRP-USDT", qty=1.0, price=100.0)
    assert buy["status"] == "filled"
    assert buy["fee"] == pytest.approx(0.1)
    assert executor.get_balance() == pytest.approx(899.9)

    positions = executor.get_positions()
    assert positions["XRP-USDT"]["quantity"] == pytest.approx(1.0)
    assert positions["XRP-USDT"]["avg_entry_price"] == pytest.approx(100.0)

    sell = executor.sell("XRP-USDT", qty=1.0, price=110.0)
    assert sell["fee"] == pytest.approx(0.11)
    assert sell["realized_pnl"] == pytest.approx(9.79)
    assert executor.get_balance() == pytest.approx(1009.79)
    assert executor.get_realized_pnl() == pytest.approx(9.79)
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

    buy = executor.buy("XRP-USDT", qty=1.0, price=200.0)
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

    executor.buy("XRP-USDT", qty=1.0, price=100.0)
    close = executor.close_position("XRP-USDT")

    # No explicit price passed, so it uses current mock market (100). Realized PnL includes entry + exit fees.
    assert close["realized_pnl"] == pytest.approx(-0.2)
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
    first.buy("XRP-USDT", qty=0.5, price=100.0)

    second = PaperExecutor(
        fetcher,
        starting_balance=5000.0,
        fee_rate=0.001,
        slippage_bps=0.0,
        db_path=str(db_path),
    )
    positions = second.get_positions()
    assert "XRP-USDT" in positions
    assert positions["XRP-USDT"]["quantity"] == pytest.approx(0.5)
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

    executor.buy("XRP-USDT", qty=0.1, price=100.0)
    executor.sell("XRP-USDT", qty=0.1, price=105.0)


def test_paper_short_open_and_cover(tmp_path: Path):
    fetcher = MockFetcher()
    executor = PaperExecutor(
        fetcher,
        starting_balance=1000.0,
        fee_rate=0.001,
        slippage_bps=0.0,
        db_path=str(tmp_path / "paper_short.db"),
    )

    short_entry = executor.sell("XRP-USDT", qty=1.0, price=100.0)
    assert short_entry["status"] == "filled"

    pos = executor.get_positions()["XRP-USDT"]
    assert pos["quantity"] == pytest.approx(-1.0)
    assert pos["side"] == "SHORT"

    cover = executor.buy("XRP-USDT", qty=1.0, price=95.0)
    assert cover["status"] == "filled"
    assert cover["realized_pnl"] is not None
    assert cover["realized_pnl"] > 0.0
    assert executor.get_positions() == {}


def test_paper_price_guard_rejects_extreme_price(tmp_path: Path):
    fetcher = MockFetcher()
    fetcher.prices["XRP-USDT"] = 2_000_000.0
    executor = PaperExecutor(
        fetcher,
        starting_balance=1000.0,
        fee_rate=0.001,
        slippage_bps=0.0,
        db_path=str(tmp_path / "paper_extreme.db"),
    )

    with pytest.raises(ValueError):
        executor.buy("XRP-USDT", qty=0.1, price=2_000_000.0)

    assert executor.get_balance() == pytest.approx(1000.0)
    assert executor.get_positions() == {}
    assert len(executor.trade_history) == 0


def test_trading_mode_validation_unknown():
    with pytest.raises(ValueError):
        TradingConfig(trading_mode="UNKNOWN")


def test_trading_mode_validation_live_guard():
    with pytest.raises(RuntimeError):
        TradingConfig(trading_mode="LIVE", allow_live_trading="NO")

    cfg = TradingConfig(trading_mode="LIVE", allow_live_trading="true")
    assert cfg.trading_mode == "LIVE"
    assert not cfg.paper_trading


def test_execution_price_matches_market_price(tmp_path: Path, patch_db):
    _ = patch_db
    fetcher = MockFetcher()
    fetcher.prices["XRP-USDT"] = 2000.0
    executor = PaperExecutor(
        fetcher,
        starting_balance=10_000.0,
        fee_rate=0.001,
        slippage_bps=0.0,
        db_path=str(tmp_path / "paper_exec_price_sync.db"),
    )

    signal = TradeSignal(
        pair="XRP-USDT",
        side="BUY",
        entry_price=1980.0,
        model_entry_price=1980.0,
        stop_price=1970.0,
        target_price=2060.0,
        confidence=0.70,
        expected_move=0.02,
        prob_up=0.7,
        prob_down=0.3,
        risk_reward=2.0,
        atr=10.0,
        volatility_regime="normal",
        market_regime="uptrend",
        btc_correlation=0.2,
        spread_pct=0.0005,
        volume_24h=1_000_000.0,
        reasoning="price sync test",
    )

    trade_id = executor.execute_signal(signal, balance=10_000.0)
    assert trade_id is not None

    positions = executor.get_positions()
    assert "XRP-USDT" in positions
    # Execution price must come from market ticker (2000), not model-derived 1980.
    assert positions["XRP-USDT"]["avg_entry_price"] == pytest.approx(2000.0)


def test_execution_price_desync_blocks_trade(tmp_path: Path, patch_db):
    _ = patch_db
    fetcher = MockFetcher()
    fetcher.prices["XRP-USDT"] = 2000.0
    executor = PaperExecutor(
        fetcher,
        starting_balance=10_000.0,
        fee_rate=0.001,
        slippage_bps=0.0,
        db_path=str(tmp_path / "paper_exec_price_desync.db"),
    )

    signal = TradeSignal(
        pair="XRP-USDT",
        side="BUY",
        entry_price=21.47,
        model_entry_price=21.47,
        stop_price=20.0,
        target_price=24.0,
        confidence=0.70,
        expected_move=0.02,
        prob_up=0.7,
        prob_down=0.3,
        risk_reward=2.0,
        atr=0.07,
        volatility_regime="normal",
        market_regime="uptrend",
        btc_correlation=0.2,
        spread_pct=0.0005,
        volume_24h=1_000_000.0,
        reasoning="desync guard test",
    )

    trade_id = executor.execute_signal(signal, balance=10_000.0)
    assert trade_id is None
    assert executor.get_positions() == {}


