"""
Risk management helpers used by the AI controller.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01
    min_position_fraction: float = 0.05
    max_position_fraction: float = 0.30
    stop_atr_multiplier: float = 1.5
    take_profit_rr: float = 2.0


class RiskManager:
    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self.cfg = cfg or RiskConfig()

    def position_size(
        self,
        balance: float,
        confidence: float,
        volatility: float,
        entry_price: float,
    ) -> float:
        confidence_scale = max(0.0, min(1.0, confidence))
        vol_scale = 1.0 / max(1.0, volatility * 100)
        fraction = self.cfg.risk_per_trade * confidence_scale * vol_scale * 10
        fraction = max(self.cfg.min_position_fraction, min(self.cfg.max_position_fraction, fraction))
        notional = balance * fraction
        if entry_price <= 0:
            return 0.0
        return max(0.0, notional / entry_price)

    def stop_loss_price(self, side: str, entry_price: float, atr: float) -> float:
        if side.upper() == "BUY":
            return entry_price - (atr * self.cfg.stop_atr_multiplier)
        return entry_price + (atr * self.cfg.stop_atr_multiplier)

    def take_profit_price(self, side: str, entry_price: float, stop_loss: float) -> float:
        risk = abs(entry_price - stop_loss)
        if side.upper() == "BUY":
            return entry_price + risk * self.cfg.take_profit_rr
        return entry_price - risk * self.cfg.take_profit_rr

