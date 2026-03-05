from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _parse_universe(raw: str) -> List[str]:
    symbols: List[str] = []
    for chunk in raw.split(","):
        symbol = chunk.strip().upper().replace("/", "-")
        if not symbol:
            continue
        if "-" not in symbol:
            symbol = f"{symbol}-USDT"
        symbols.append(symbol)
    deduped: List[str] = []
    seen = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


@dataclass(frozen=True)
class QuantEngineConfig:
    universe: List[str] = field(
        default_factory=lambda: _parse_universe(
            _env_str(
                "UNIVERSE",
                "BTC-USDT,ETH-USDT,SOL-USDT,BNB-USDT,DOGE-USDT,XRP-USDT,AVAX-USDT",
            )
        )
    )
    auto_expand_universe: bool = _env_bool("UNIVERSE_AUTO_EXPAND", True)
    max_universe_size: int = _env_int("MAX_UNIVERSE_SIZE", 14)

    # Cadence
    market_interval_sec: int = _env_int("ENGINE_MARKET_INTERVAL_SECONDS", 10)
    signal_interval_sec: int = _env_int("ENGINE_SIGNAL_INTERVAL_SECONDS", 60)
    rebalance_interval_sec: int = _env_int("ENGINE_REBALANCE_INTERVAL_SECONDS", 300)
    dashboard_refresh_sec: int = _env_int("DASHBOARD_REFRESH_SECONDS", 5)

    # Risk and execution
    initial_balance: float = _env_float("PAPER_INITIAL_BALANCE", 10000.0)
    fee_rate: float = _env_float("PAPER_FEE_RATE", 0.001)
    slippage_bps: float = _env_float("PAPER_SLIPPAGE_BPS", 2.0)
    partial_fill_min: float = _env_float("PAPER_PARTIAL_FILL_MIN", 0.6)
    partial_fill_max: float = _env_float("PAPER_PARTIAL_FILL_MAX", 1.0)
    max_daily_loss_pct: float = _env_float("MAX_DAILY_LOSS_PCT", 0.08)
    max_drawdown_pct: float = _env_float("MAX_DRAWDOWN_PCT", 0.20)
    max_exposure_pct: float = _env_float("MAX_EXPOSURE_PCT", 0.90)
    max_simultaneous_trades: int = _env_int("MAX_SIMULTANEOUS_TRADES", 40)
    base_risk_factor: float = _env_float("RISK_PER_TRADE", 0.0075)
    enable_shorts: bool = _env_bool("ALLOW_SHORTS", True)

    # Strategy and RL
    strategy_discovery_samples: int = _env_int("STRATEGY_DISCOVERY_SAMPLES", 36)
    strategy_discovery_lookback: int = _env_int("STRATEGY_DISCOVERY_LOOKBACK", 240)
    min_signal_confidence: float = _env_float("CONFIDENCE_THRESHOLD", 0.53)
    rl_learning_rate: float = _env_float("RL_LEARNING_RATE", 0.08)
    rl_discount_factor: float = _env_float("RL_DISCOUNT_FACTOR", 0.95)
    rl_exploration: float = _env_float("RL_EXPLORATION", 0.12)
    trade_target_min: int = _env_int("TARGET_TRADES_MIN", 100)
    trade_target_max: int = _env_int("TARGET_TRADES_MAX", 300)

    def normalized(self) -> "QuantEngineConfig":
        universe = self.universe or ["BTC-USDT"]
        partial_min = min(max(self.partial_fill_min, 0.05), 1.0)
        partial_max = min(max(self.partial_fill_max, partial_min), 1.0)
        return QuantEngineConfig(
            universe=universe[: max(1, self.max_universe_size)],
            auto_expand_universe=self.auto_expand_universe,
            max_universe_size=max(1, self.max_universe_size),
            market_interval_sec=max(2, self.market_interval_sec),
            signal_interval_sec=max(10, self.signal_interval_sec),
            rebalance_interval_sec=max(self.signal_interval_sec, self.rebalance_interval_sec),
            dashboard_refresh_sec=max(2, self.dashboard_refresh_sec),
            initial_balance=max(100.0, self.initial_balance),
            fee_rate=max(0.0, self.fee_rate),
            slippage_bps=max(0.0, self.slippage_bps),
            partial_fill_min=partial_min,
            partial_fill_max=partial_max,
            max_daily_loss_pct=min(max(self.max_daily_loss_pct, 0.01), 0.99),
            max_drawdown_pct=min(max(self.max_drawdown_pct, 0.05), 0.99),
            max_exposure_pct=min(max(self.max_exposure_pct, 0.1), 1.5),
            max_simultaneous_trades=max(1, self.max_simultaneous_trades),
            base_risk_factor=min(max(self.base_risk_factor, 0.0005), 0.10),
            enable_shorts=self.enable_shorts,
            strategy_discovery_samples=max(10, self.strategy_discovery_samples),
            strategy_discovery_lookback=max(80, self.strategy_discovery_lookback),
            min_signal_confidence=min(max(self.min_signal_confidence, 0.1), 0.95),
            rl_learning_rate=min(max(self.rl_learning_rate, 0.001), 0.5),
            rl_discount_factor=min(max(self.rl_discount_factor, 0.1), 0.999),
            rl_exploration=min(max(self.rl_exploration, 0.0), 0.5),
            trade_target_min=max(1, self.trade_target_min),
            trade_target_max=max(self.trade_target_min, self.trade_target_max),
        )
