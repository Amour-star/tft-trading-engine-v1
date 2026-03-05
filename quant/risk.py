from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import numpy as np

from quant.config import QuantEngineConfig
from quant.types import PortfolioState, RiskDecision


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


@dataclass
class RiskStateRuntime:
    day: str
    start_equity: float
    peak_equity: float
    daily_realized_pnl: float
    trading_paused: bool = False


class RiskManager:
    """Institutional risk constraints and circuit breakers."""

    def __init__(self, cfg: QuantEngineConfig, initial_equity: float) -> None:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        self.cfg = cfg
        self.state = RiskStateRuntime(
            day=today,
            start_equity=max(1.0, initial_equity),
            peak_equity=max(1.0, initial_equity),
            daily_realized_pnl=0.0,
            trading_paused=False,
        )

    def update_account_state(self, portfolio: PortfolioState) -> None:
        now_day = portfolio.timestamp.strftime("%Y-%m-%d")
        if now_day != self.state.day:
            self.state = RiskStateRuntime(
                day=now_day,
                start_equity=max(1.0, portfolio.equity),
                peak_equity=max(1.0, portfolio.equity),
                daily_realized_pnl=0.0,
                trading_paused=False,
            )
            return
        self.state.peak_equity = max(self.state.peak_equity, _safe_float(portfolio.equity, 0.0))

        drawdown = self.current_drawdown(portfolio.equity)
        if drawdown >= self.cfg.max_drawdown_pct:
            self.state.trading_paused = True

        daily_loss_pct = self.current_daily_loss_pct(portfolio.equity)
        if daily_loss_pct >= self.cfg.max_daily_loss_pct:
            self.state.trading_paused = True

    def register_closed_trade(self, realized_pnl: float) -> None:
        self.state.daily_realized_pnl += _safe_float(realized_pnl, 0.0)

    def current_drawdown(self, equity: float) -> float:
        peak = max(self.state.peak_equity, 1e-9)
        return max(0.0, (peak - _safe_float(equity, 0.0)) / peak)

    def current_daily_loss_pct(self, equity: float) -> float:
        start = max(self.state.start_equity, 1e-9)
        return max(0.0, (start - _safe_float(equity, 0.0)) / start)

    def evaluate_new_risk(
        self,
        symbol: str,
        requested_notional: float,
        exposure_notional: float,
        open_positions: int,
        equity: float,
    ) -> RiskDecision:
        if self.state.trading_paused:
            return RiskDecision(accepted=False, reason="trading_paused", max_allowed_notional=0.0)
        if open_positions >= self.cfg.max_simultaneous_trades:
            return RiskDecision(accepted=False, reason="max_simultaneous_trades", max_allowed_notional=0.0)
        if equity <= 0:
            return RiskDecision(accepted=False, reason="invalid_equity", max_allowed_notional=0.0)

        max_exposure_notional = equity * self.cfg.max_exposure_pct
        available = max(0.0, max_exposure_notional - exposure_notional)
        if available <= 0:
            return RiskDecision(
                accepted=False,
                reason="max_exposure_reached",
                max_allowed_notional=0.0,
            )

        allowed = min(available, equity * 0.30)
        accepted = requested_notional <= allowed and requested_notional > 0.0
        risk_factor = min(1.0, allowed / max(requested_notional, 1e-9))
        return RiskDecision(
            accepted=accepted,
            reason="ok" if accepted else "requested_notional_too_large",
            risk_factor=risk_factor,
            max_allowed_notional=allowed,
        )

    def status_payload(self) -> Dict[str, float | str | bool]:
        return {
            "day": self.state.day,
            "start_equity": self.state.start_equity,
            "peak_equity": self.state.peak_equity,
            "daily_realized_pnl": self.state.daily_realized_pnl,
            "trading_paused": self.state.trading_paused,
        }
