"""PnL helper functions for deterministic trade accounting."""
from __future__ import annotations


def calculate_realized_pnl(
    *,
    entry_price: float,
    exit_price: float,
    quantity: float,
    side: str,
) -> float:
    """Calculate realized PnL for spot-style long/short accounting."""
    entry = float(entry_price)
    exit_ = float(exit_price)
    qty = float(quantity)
    direction = str(side or "BUY").upper()

    if direction == "BUY":
        return (exit_ - entry) * qty
    if direction == "SELL":
        return (entry - exit_) * qty
    raise ValueError(f"Unsupported side for PnL calculation: {side}")
