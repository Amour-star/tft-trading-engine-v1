import pytest

from utils.pnl import calculate_realized_pnl


def test_calculate_realized_pnl_buy():
    pnl = calculate_realized_pnl(
        entry_price=0.5,
        exit_price=0.6,
        quantity=100.0,
        side="BUY",
    )
    assert pnl == pytest.approx(10.0)


def test_calculate_realized_pnl_sell():
    pnl = calculate_realized_pnl(
        entry_price=0.6,
        exit_price=0.5,
        quantity=100.0,
        side="SELL",
    )
    assert pnl == pytest.approx(10.0)
