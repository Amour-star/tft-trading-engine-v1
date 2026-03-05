from __future__ import annotations

import asyncio
import contextlib
import math
import signal
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

from loguru import logger

from data.database import DecisionEvent, EngineState, SignalRecord, get_session, init_db
from quant.analytics import PerformanceAnalyticsEngine
from quant.config import QuantEngineConfig
from quant.event_bus import AsyncEventBus
from quant.execution import ExecutionEngine
from quant.features import FeatureEngineeringEngine
from quant.market_data import MarketDataEngine
from quant.portfolio import PortfolioOptimizer
from quant.regime import MarketRegimeAI
from quant.rl_trader import ReinforcementLearningTrader
from quant.risk import RiskManager
from quant.strategy import StrategyEngine
from quant.types import RegimeState, StrategySignal


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


class QuantTradingOrchestrator:
    """Async, event-driven multi-asset trading system."""

    def __init__(self, cfg: Optional[QuantEngineConfig] = None) -> None:
        self.cfg = (cfg or QuantEngineConfig()).normalized()
        init_db()

        self.event_bus = AsyncEventBus()
        self.market_data = MarketDataEngine(self.cfg)
        self.features = FeatureEngineeringEngine()
        self.regime_ai = MarketRegimeAI()
        self.strategy = StrategyEngine(self.cfg)
        self.rl = ReinforcementLearningTrader(self.cfg)
        self.portfolio_optimizer = PortfolioOptimizer()
        self.execution = ExecutionEngine(self.cfg)
        initial_state = self.execution.portfolio_state(prices={})
        self.risk = RiskManager(self.cfg, initial_equity=initial_state.equity)
        self.analytics = PerformanceAnalyticsEngine()
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._last_market_ts = 0.0
        self._last_signal_ts = 0.0
        self._last_rebalance_ts = 0.0
        self._rl_open_context: Dict[str, Tuple[StrategySignal, RegimeState, bool, str]] = {}

    def request_stop(self) -> None:
        self._stop.set()

    def bind_signals(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                signal.signal(sig, lambda *_: self.request_stop())

    async def run(self) -> None:
        self.bind_signals()
        self._set_engine_state("running", {"value": True, "component": "quant_orchestrator"})
        self._set_engine_state("paused", {"value": False})
        await self.event_bus.publish(
            "ENGINE_STARTED",
            {
                "universe_size": len(self.cfg.universe),
                "market_interval_sec": self.cfg.market_interval_sec,
                "signal_interval_sec": self.cfg.signal_interval_sec,
                "rebalance_interval_sec": self.cfg.rebalance_interval_sec,
            },
        )

        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now - self._last_market_ts >= self.cfg.market_interval_sec:
                    await self._run_market_cycle()
                    self._last_market_ts = now

                if now - self._last_signal_ts >= self.cfg.signal_interval_sec:
                    await self._run_signal_cycle()
                    self._last_signal_ts = now

                if now - self._last_rebalance_ts >= self.cfg.rebalance_interval_sec:
                    await self._run_rebalance_cycle()
                    self._last_rebalance_ts = now

                await asyncio.sleep(0.25)
        finally:
            self._set_engine_state("running", {"value": False, "component": "quant_orchestrator"})
            await self.event_bus.publish("ENGINE_STOPPED", {"reason": "shutdown"})

    async def _run_market_cycle(self) -> None:
        snapshots = await self.market_data.update_all()
        await self.event_bus.publish(
            "MARKET_DATA_UPDATE",
            {"symbols": len(snapshots), "timestamp": datetime.utcnow().isoformat()},
        )

    async def _run_signal_cycle(self) -> None:
        snapshots = dict(self.market_data.snapshots)
        if not snapshots:
            await self.event_bus.publish("SIGNAL_SKIPPED", {"reason": "no_market_data"})
            return

        prices = {symbol: snap.ticker_price for symbol, snap in snapshots.items()}
        before_state = self.execution.portfolio_state(prices)
        self.risk.update_account_state(before_state)
        if self.risk.state.trading_paused:
            await self.event_bus.publish(
                "RISK_REJECTED",
                {
                    "reason": "trading_paused",
                    "drawdown": self.risk.current_drawdown(before_state.equity),
                    "daily_loss_pct": self.risk.current_daily_loss_pct(before_state.equity),
                },
            )
            return

        signals: Dict[str, StrategySignal] = {}
        regimes: Dict[str, RegimeState] = {}
        packets = {}

        for symbol, snapshot in snapshots.items():
            packet = self.features.compute(snapshot, timeframe="1min")
            regime = self.regime_ai.classify(packet)
            self.strategy.discover_for_symbol(symbol, snapshot)
            signal_candidate = self.strategy.generate_signal(snapshot, packet, regime)
            has_position = any(lot.symbol == symbol for lot in self.execution.open_lots.values())
            rl_decision = self.rl.decide(signal_candidate, regime, has_position)

            threshold = max(
                0.05,
                min(
                    0.95,
                    self.cfg.min_signal_confidence + regime.threshold_shift - rl_decision.confidence_boost,
                ),
            )
            signal_candidate.confidence = max(0.0, min(1.0, signal_candidate.confidence + rl_decision.confidence_boost))

            await self._persist_signal(signal_candidate, regime, threshold)
            await self.event_bus.publish(
                "SIGNAL_GENERATED",
                {
                    "symbol": symbol,
                    "direction": signal_candidate.direction,
                    "confidence": round(signal_candidate.confidence, 6),
                    "score": round(signal_candidate.score, 6),
                    "strategy": signal_candidate.strategy_name,
                    "regime": regime.label,
                },
            )

            if signal_candidate.confidence < threshold:
                continue
            if signal_candidate.direction == 0:
                continue
            signals[symbol] = signal_candidate
            regimes[symbol] = regime
            packets[symbol] = packet

        if not signals:
            after_state = self.execution.portfolio_state(prices)
            self.risk.update_account_state(after_state)
            metrics = self.analytics.update(after_state)
            self._set_engine_state(
                "quant_runtime",
                {
                    "equity": float(after_state.equity),
                    "balance": float(after_state.balance),
                    "realized_pnl": float(after_state.realized_pnl),
                    "unrealized_pnl": float(after_state.unrealized_pnl),
                    "open_positions": int(len(after_state.open_positions)),
                    "risk": self.risk.status_payload(),
                    "metrics": metrics,
                    "updated_at": datetime.utcnow().isoformat(),
                },
            )
            self._persist_decision_event(status="no_trade", reason="threshold_or_hold")
            return

        targets = self.portfolio_optimizer.optimize(
            total_equity=max(1.0, before_state.equity),
            signals=signals,
            features=packets,
            risk_factor=self.cfg.base_risk_factor,
        )
        target_by_symbol = {t.symbol: t for t in targets}

        trade_reports = []
        for symbol, signal_item in signals.items():
            regime = regimes[symbol]
            target = target_by_symbol.get(symbol)
            if target is None:
                continue
            risk_check = self.risk.evaluate_new_risk(
                symbol=symbol,
                requested_notional=target.target_notional,
                exposure_notional=before_state.exposure_notional,
                open_positions=len(before_state.open_positions),
                equity=before_state.equity,
            )
            if not risk_check.accepted:
                await self.event_bus.publish(
                    "RISK_REJECTED",
                    {
                        "symbol": symbol,
                        "reason": risk_check.reason,
                        "requested_notional": target.target_notional,
                        "max_allowed_notional": risk_check.max_allowed_notional,
                    },
                )
                continue

            snapshot = snapshots[symbol]
            reports = self.execution.process_signal(
                signal=signal_item,
                regime=regime,
                mark_price=snapshot.ticker_price,
                spread_pct=snapshot.spread_pct,
                max_notional=min(target.target_notional, risk_check.max_allowed_notional),
            )
            for report in reports:
                trade_reports.append(report)
                if report.status == "opened":
                    has_position = True
                    action = "enter"
                    self._rl_open_context[report.trade_id] = (signal_item, regime, has_position, action)
                if report.status == "closed":
                    self.risk.register_closed_trade(report.realized_pnl)
                    context = self._rl_open_context.pop(report.trade_id, None)
                    if context is not None:
                        prev_signal, prev_regime, prev_has_position, action = context
                        reward = report.realized_pnl / max(before_state.equity, 1.0)
                        self.rl.learn(
                            previous_signal=prev_signal,
                            previous_regime=prev_regime,
                            previous_has_position=prev_has_position,
                            action=action,
                            reward=reward,
                            next_signal=signal_item,
                            next_regime=regime,
                            next_has_position=False,
                        )

        after_state = self.execution.portfolio_state(prices)
        self.risk.update_account_state(after_state)
        metrics = self.analytics.update(after_state)

        await self.event_bus.publish(
            "EXECUTION_COMPLETE",
            {
                "reports": len(trade_reports),
                "open_positions": len(after_state.open_positions),
                "equity": round(after_state.equity, 6),
                "sharpe": round(_safe_float(metrics.get("sharpe_ratio"), 0.0), 6),
                "win_rate": round(_safe_float(metrics.get("win_rate"), 0.0), 6),
            },
        )
        self._set_engine_state(
            "quant_runtime",
            {
                "equity": float(after_state.equity),
                "balance": float(after_state.balance),
                "realized_pnl": float(after_state.realized_pnl),
                "unrealized_pnl": float(after_state.unrealized_pnl),
                "open_positions": int(len(after_state.open_positions)),
                "risk": self.risk.status_payload(),
                "metrics": metrics,
                "updated_at": datetime.utcnow().isoformat(),
            },
        )
        self._persist_decision_event(
            status="trade_opened" if any(r.status == "opened" for r in trade_reports) else "no_trade",
            reason="signal_cycle_complete",
        )

    async def _run_rebalance_cycle(self) -> None:
        snapshots = dict(self.market_data.snapshots)
        if not snapshots:
            return
        prices = {symbol: snap.ticker_price for symbol, snap in snapshots.items()}
        state = self.execution.portfolio_state(prices)
        force_rebalance = state.exposure_notional > state.equity * self.cfg.max_exposure_pct
        now = datetime.utcnow()
        for position in list(state.open_positions):
            price = _safe_float(prices.get(position.symbol), position.avg_entry_price)
            age_sec = (now - position.opened_at).total_seconds()
            adverse = (price - position.avg_entry_price) * position.quantity < 0
            stale_risk = age_sec >= 1800 and adverse
            if not (force_rebalance or stale_risk):
                continue
            reports = self.execution.close_by_risk(
                symbol=position.symbol,
                price=price,
                reason="rebalance",
            )
            if reports:
                await self.event_bus.publish(
                    "REBALANCE_EXECUTED",
                    {
                        "symbol": position.symbol,
                        "closed_trades": len(reports),
                    },
                )

    async def _persist_signal(self, signal: StrategySignal, regime: RegimeState, threshold: float) -> None:
        session = get_session()
        try:
            session.add(
                SignalRecord(
                    timestamp=signal.timestamp,
                    pair=signal.symbol,
                    timeframe="1min",
                    decision="LONG" if signal.direction > 0 else ("SHORT" if signal.direction < 0 else "HOLD"),
                    side="BUY" if signal.direction > 0 else ("SELL" if signal.direction < 0 else "HOLD"),
                    signal_score=float(signal.score),
                    momentum_factor=float(signal.components.get("momentum_breakout", 0.0)),
                    volatility_factor=float(signal.components.get("volatility_breakout", 0.0)),
                    trend_factor=float(signal.components.get("momentum_breakout", 0.0)),
                    mean_reversion_factor=float(signal.components.get("mean_reversion", 0.0)),
                    volume_factor=float(signal.components.get("orderflow_imbalance", 0.0)),
                    volume_imbalance=float(signal.components.get("orderflow_imbalance", 0.0)),
                    model_confidence=float(signal.confidence),
                    regime=regime.label,
                    payload={
                        "strategy": signal.strategy_name,
                        "reason": signal.reason,
                        "threshold": threshold,
                        "regime_confidence": regime.confidence,
                    },
                )
            )
            session.commit()
        finally:
            session.close()

    def _persist_decision_event(self, status: str, reason: str) -> None:
        session = get_session()
        try:
            session.add(
                DecisionEvent(
                    timestamp=datetime.utcnow(),
                    mode="PAPER",
                    status=status,
                    reason=reason,
                    candidates_evaluated=max(1, len(self.market_data.snapshots)),
                    candidates_valid=max(0, len(self.execution.open_lots)),
                    best_pair=None,
                    best_score=None,
                    best_ai_score=None,
                    best_confidence=None,
                    best_prob_up=None,
                    best_prob_down=None,
                    regime="multi",
                    volatility_regime="dynamic",
                    adaptive_threshold=self.cfg.min_signal_confidence,
                    top_candidates_json=[],
                )
            )
            session.commit()
        finally:
            session.close()

    def _set_engine_state(self, key: str, payload: Dict[str, object]) -> None:
        session = get_session()
        try:
            row = session.query(EngineState).filter(EngineState.key == key).first()
            if row:
                row.value = dict(payload)
                row.updated_at = datetime.utcnow()
            else:
                session.add(EngineState(key=key, value=dict(payload)))
            session.commit()
        finally:
            session.close()
