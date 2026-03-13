"""
Hard deterministic trade validator for paper and live execution.
"""
from __future__ import annotations

import os
from typing import Any, Dict

from loguru import logger

from config.settings import settings


VALIDATION_TAG = "[VALIDATION]"


def validate_trade(trade_data: Dict[str, Any], account_balance: float) -> Dict[str, Any]:
    """Apply hard thresholds that must pass before executing any trade."""
    symbol = str(trade_data.get("symbol", "")).upper()
    entry_price = float(trade_data.get("entry_price") or 0.0)
    mark_price = float(trade_data.get("mark_price") or 0.0)
    quantity = float(trade_data.get("quantity") or 0.0)
    confidence = float(trade_data.get("confidence") or 0.0)
    signal_score = float(trade_data.get("signal_score") or 0.0)
    expected_edge_pct = float(trade_data.get("expected_edge_pct") or 0.0)
    estimated_fee_drag_pct = float(trade_data.get("estimated_fee_drag_pct") or 0.0)
    risk_per_trade = float(
        trade_data.get("risk_per_trade", settings.trading.risk_per_trade or 0.0)
    )
    full_balance_mode = bool(trade_data.get("full_balance_mode", False))
    default_position_fraction = 0.99 if full_balance_mode else float(
        os.getenv("POSITION_MAX_BALANCE_FRACTION", "0.50")
    )
    max_position_fraction = float(
        trade_data.get("max_position_fraction", default_position_fraction)
    )
    max_position_fraction = max(0.01, min(1.0, max_position_fraction))

    if entry_price <= 0:
        reason = "Entry price must be greater than zero."
        logger.error(f"{VALIDATION_TAG} {symbol} rejected: {reason} entry_price={entry_price:.8f}")
        return {"valid": False, "reason": reason}

    if mark_price <= 0:
        reason = "Mark price is unavailable or invalid."
        logger.error(f"{VALIDATION_TAG} {symbol} rejected: {reason}")
        return {"valid": False, "reason": reason}

    if quantity <= 0:
        reason = "Quantity must be positive."
        logger.error(f"{VALIDATION_TAG} {symbol} rejected: {reason}")
        return {"valid": False, "reason": reason}

    max_position_value = account_balance * max_position_fraction
    position_value = entry_price * quantity
    if position_value > max_position_value:
        pct_label = int(max_position_fraction * 100)
        reason = f"Position value exceeds {pct_label}% of account balance."
        logger.error(
            f"{VALIDATION_TAG} {symbol} rejected: {reason} "
            f"(position_value={position_value:.2f} limit={max_position_value:.2f})"
        )
        return {"valid": False, "reason": reason}

    if not full_balance_mode and risk_per_trade > 0.02:
        reason = "Risk per trade cannot exceed 2%."
        logger.error(
            f"{VALIDATION_TAG} {symbol} rejected: {reason} risk_per_trade={risk_per_trade:.4f}"
        )
        return {"valid": False, "reason": reason}

    aggression = max(0.1, float(settings.trading.aggression_level))
    required_confidence = max(0.40, float(settings.trading.confidence_threshold) / aggression)
    if confidence < required_confidence:
        reason = "Confidence below configured threshold."
        logger.warning(
            f"{VALIDATION_TAG} {symbol} rejected: {reason} "
            f"(confidence={confidence:.3f} threshold={required_confidence:.3f})"
        )
        return {"valid": False, "reason": reason}

    min_signal_score = max(0.45, required_confidence - 0.05)
    if signal_score > 0.0 and signal_score < min_signal_score:
        reason = "Signal score below execution threshold."
        logger.warning(
            f"{VALIDATION_TAG} {symbol} rejected: {reason} "
            f"(signal_score={signal_score:.3f} threshold={min_signal_score:.3f})"
        )
        return {"valid": False, "reason": reason}

    if expected_edge_pct != 0.0 or estimated_fee_drag_pct != 0.0:
        if expected_edge_pct <= max(0.0, estimated_fee_drag_pct):
            reason = "Expected edge does not clear fees and slippage."
            logger.warning(
                f"{VALIDATION_TAG} {symbol} rejected: {reason} "
                f"(edge={expected_edge_pct:.6f} costs={estimated_fee_drag_pct:.6f})"
            )
            return {"valid": False, "reason": reason}

    risk_amount = account_balance * risk_per_trade
    if entry_price <= 0 or risk_amount <= 0:
        reason = "Risk sizing could not be computed."
        logger.error(f"{VALIDATION_TAG} {symbol} rejected: {reason}")
        return {"valid": False, "reason": reason}

    stop_price = float(trade_data.get("stop_price") or 0.0)
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        reason = "Stop price is required for risk sizing."
        logger.error(f"{VALIDATION_TAG} {symbol} rejected: {reason}")
        return {"valid": False, "reason": reason}

    max_quantity = max_position_value / entry_price
    if full_balance_mode:
        if quantity > max_quantity:
            reason = "Quantity exceeds full-balance position cap."
            logger.error(
                f"{VALIDATION_TAG} {symbol} rejected: {reason} qty={quantity:.8f} "
                f"cap_qty={max_quantity:.8f}"
            )
            return {"valid": False, "reason": reason}
    else:
        risk_quantity = risk_amount / stop_distance
        if quantity > risk_quantity or quantity > max_quantity:
            reason = "Quantity exceeds risk or position caps."
            logger.error(
                f"{VALIDATION_TAG} {symbol} rejected: {reason} qty={quantity:.8f} "
                f"risk_qty={risk_quantity:.8f} cap_qty={max_quantity:.8f}"
            )
            return {"valid": False, "reason": reason}

    price_ratio = abs(mark_price - entry_price) / entry_price
    if price_ratio > 0.5:
        reason = "Scale mismatch between entry price and mark price."
        logger.warning(
            f"{VALIDATION_TAG} {symbol} rejected: {reason} ratio={price_ratio:.3f}"
        )
        return {"valid": False, "reason": reason}

    return {"valid": True, "reason": "Passed deterministic safety checks."}
