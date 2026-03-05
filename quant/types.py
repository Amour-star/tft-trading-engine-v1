from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


@dataclass
class MarketSnapshot:
    symbol: str
    timestamp: datetime
    ticker_price: float
    best_bid: float
    best_ask: float
    spread_pct: float
    orderbook_imbalance: float
    volume_imbalance: float
    funding_rate: float
    realized_volatility: float
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class FeaturePacket:
    symbol: str
    timestamp: datetime
    timeframe: str
    raw_features: Dict[str, float]
    normalized_features: Dict[str, float]


@dataclass
class RegimeState:
    symbol: str
    timestamp: datetime
    label: str
    confidence: float
    position_size_mult: float
    threshold_shift: float
    aggressiveness_mult: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySignal:
    symbol: str
    timestamp: datetime
    direction: int  # -1 short, 0 flat, +1 long
    confidence: float
    score: float
    strategy_name: str
    regime: str
    reason: str
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioTarget:
    symbol: str
    target_weight: float
    target_notional: float
    risk_budget: float


@dataclass
class ExecutionReport:
    trade_id: str
    symbol: str
    side: str
    status: str
    filled_qty: float
    avg_price: float
    fee_paid: float
    slippage_bps: float
    realized_pnl: float
    message: str = ""


@dataclass
class PositionSnapshot:
    symbol: str
    quantity: float
    avg_entry_price: float
    stop_price: float
    take_profit: float
    trailing_stop: float
    opened_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskDecision:
    accepted: bool
    reason: str
    risk_factor: float = 0.0
    max_allowed_notional: float = 0.0


@dataclass
class PortfolioState:
    timestamp: datetime
    balance: float
    realized_pnl: float
    unrealized_pnl: float
    equity: float
    open_positions: List[PositionSnapshot] = field(default_factory=list)
    exposure_notional: float = 0.0
