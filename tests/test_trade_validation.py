"""
Tests for the deterministic trade validator.
"""

from services.trade_validator import validate_trade


def _base_trade():
    return {
        "symbol": "PEPE-USDT",
        "entry_price": 10.0,
        "stop_price": 9.5,
        "target_price": 12.0,
        "confidence": 0.8,
        "mark_price": 10.1,
        "quantity": 1.0,
        "risk_per_trade": 0.01,
    }


def test_entry_price_zero_rejected():
    trade = _base_trade()
    trade["entry_price"] = 0.0
    result = validate_trade(trade, account_balance=1_000.0)
    assert result["valid"] is False
    assert "Entry price" in result["reason"]


def test_mark_price_scale_mismatch_blocks():
    trade = _base_trade()
    trade["mark_price"] = 30.0
    trade["entry_price"] = 10.0
    result = validate_trade(trade, account_balance=1_000.0)
    assert result["valid"] is False
    assert "Scale mismatch" in result["reason"]


def test_quantity_overrisk_rejected():
    trade = _base_trade()
    trade["quantity"] = 500.0
    result = validate_trade(trade, account_balance=1_000.0)
    assert result["valid"] is False
    assert "Position value" in result["reason"]


def test_negative_mark_price_rejected():
    trade = _base_trade()
    trade["mark_price"] = -10.0
    result = validate_trade(trade, account_balance=1_000.0)
    assert result["valid"] is False
    assert "Mark price" in result["reason"]


def test_normal_trade_passes():
    trade = _base_trade()
    trade["quantity"] = 2.0
    result = validate_trade(trade, account_balance=10_000.0)
    assert result["valid"] is True
