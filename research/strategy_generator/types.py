from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


@dataclass
class StrategyCandidate:
    strategy_id: str
    symbol: str
    timeframe: str
    indicators: List[str]
    entry_logic: Dict[str, Dict[str, Any]]
    filter_logic: Dict[str, Dict[str, Any]]
    min_confirmations: int
    allow_short: bool
    max_hold_bars: int
    stop_atr_multiplier: float
    take_atr_multiplier: float
    trailing_atr_multiplier: float
    risk_per_trade: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "StrategyCandidate":
        return cls(
            strategy_id=str(payload["strategy_id"]),
            symbol=str(payload["symbol"]),
            timeframe=str(payload.get("timeframe", "15min")),
            indicators=[str(item) for item in payload.get("indicators", [])],
            entry_logic=dict(payload.get("entry_logic", {})),
            filter_logic=dict(payload.get("filter_logic", {})),
            min_confirmations=int(payload.get("min_confirmations", 1)),
            allow_short=bool(payload.get("allow_short", True)),
            max_hold_bars=int(payload.get("max_hold_bars", 12)),
            stop_atr_multiplier=float(payload.get("stop_atr_multiplier", 1.5)),
            take_atr_multiplier=float(payload.get("take_atr_multiplier", 2.0)),
            trailing_atr_multiplier=float(payload.get("trailing_atr_multiplier", 1.2)),
            risk_per_trade=float(payload.get("risk_per_trade", 0.01)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class BacktestTrade:
    strategy_id: str
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    return_pct: float
    hold_bars: int
    exit_reason: str
    fees_paid: float
    spread_cost: float
    slippage_cost: float


@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    avg_trade_return: float = 0.0
    expectancy: float = 0.0
    passed: bool = False
    score: float = 0.0
    failure_reason: str = ""
    equity_curve: List[float] = field(default_factory=list)
    trade_returns: List[float] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["trades"] = [
            {
                **asdict(trade),
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
            }
            for trade in self.trades
        ]
        return _json_safe(payload)


@dataclass
class WalkForwardFold:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics

    def to_payload(self) -> Dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_metrics": self.train_metrics.to_payload(),
            "test_metrics": self.test_metrics.to_payload(),
        }


@dataclass
class StrategyEvaluation:
    candidate: StrategyCandidate
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    walk_forward_folds: List[WalkForwardFold] = field(default_factory=list)
    selected: bool = False
    deployed: bool = False
    rank_percentile: float = 1.0
    evaluation_time: datetime = field(default_factory=datetime.utcnow)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_payload(),
            "train_metrics": self.train_metrics.to_payload(),
            "test_metrics": self.test_metrics.to_payload(),
            "walk_forward_folds": [fold.to_payload() for fold in self.walk_forward_folds],
            "selected": self.selected,
            "deployed": self.deployed,
            "rank_percentile": self.rank_percentile,
            "evaluation_time": self.evaluation_time.isoformat(),
        }


@dataclass
class ResearchRunSummary:
    run_id: str
    symbol: str
    timeframe: str
    candidate_count: int
    accepted_count: int
    selected_count: int
    deployed_count: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    notes: Dict[str, Any] = field(default_factory=dict)
