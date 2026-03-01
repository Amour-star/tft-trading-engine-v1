"""
Abstract execution interface and shared trade pipeline.
"""
from __future__ import annotations

import json
import math
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger

from config.settings import XRP_ONLY_SYMBOL, settings
from data.database import PositionEvent
from data.database import Trade, get_session
from risk.safety_layer import (
    abnormal_trade_detector,
    is_safe_mode,
    load_limits,
    validate_position_size,
)
from services.trade_validator import validate_trade

try:
    from services.ai_auditor import AUDITOR_READY, audit_trade
except ImportError:
    AUDITOR_READY = False

    def audit_trade(*args, **kwargs):
        return {
            "status": "auditor_disabled",
            "confidence_adjustment": 0,
            "risk_override": False,
        }
from utils.logging import log_trade

if TYPE_CHECKING:
    from data.fetcher import KuCoinDataFetcher
    from engine.decision import TradeSignal


class BaseExecutor(ABC):
    """Shared executor API for both LIVE and PAPER modes."""

    mode: str = "BASE"

    def __init__(self, fetcher: "KuCoinDataFetcher") -> None:
        self.fetcher = fetcher
        self._symbol_cache: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def buy(
        self,
        symbol: str,
        qty: float,
        price: Optional[float] = None,
        market_ticker: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def sell(self, symbol: str, qty: float, price: Optional[float] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def close_position(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError

    def force_close(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        side: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Force close an open position immediately.
        If side is SELL, close by buying back (short cover).
        """
        if str(side or "BUY").upper() == "SELL":
            return self.buy(symbol, quantity, price=price)
        return self.sell(symbol, quantity, price=price)

    @abstractmethod
    def get_positions(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_balance(self) -> float:
        raise NotImplementedError

    def place_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
    ) -> Dict[str, Any]:
        return {
            "status": "skipped",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "stop_price": stop_price,
        }

    def place_limit_sell(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        return {"status": "skipped", "symbol": symbol, "quantity": quantity, "price": price}

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> Dict[str, Any]:
        if str(side).lower() == "sell":
            return self.place_limit_sell(symbol, quantity, price)
        return {
            "status": "skipped",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
        }

    def cancel_order(self, order_id: str) -> bool:
        _ = order_id
        return False

    def cancel_all_orders(self, symbol: str) -> None:
        _ = symbol

    def reconcile_positions(self) -> None:
        """Optional startup reconciliation hook."""

    def emit_position_event(
        self,
        event_type: str,
        *,
        symbol: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> PositionEvent:
        from services.reconciliation import emit_position_event

        return emit_position_event(
            event_type=event_type,
            symbol=symbol,
            details=details,
        )

    def schedule_reconciliation(
        self,
        symbols: list[str],
        *,
        source: str = "manual",
    ) -> list[PositionEvent]:
        from services.reconciliation import schedule_reconciliation

        _ = self
        return schedule_reconciliation(symbols, source=source)

    def reload_runtime_state(self) -> None:
        """Optional hook to reload local runtime snapshots after external resets."""

    def on_execution_failed(self, symbol: str, quantity: float) -> None:
        _ = symbol
        _ = quantity

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self._symbol_cache:
            self._symbol_cache[symbol] = self.fetcher.get_symbol_info(symbol)
        return self._symbol_cache[symbol]

    def round_price(self, price: float, symbol: str) -> float:
        info = self.get_symbol_info(symbol)
        increment = info["price_increment"]
        if increment <= 0:
            return price
        decimals = max(0, -int(math.floor(math.log10(increment))))
        return round(math.floor(price / increment) * increment, decimals)

    def round_quantity(self, qty: float, symbol: str) -> float:
        info = self.get_symbol_info(symbol)
        increment = info["base_increment"]
        if increment <= 0:
            return qty
        decimals = max(0, -int(math.floor(math.log10(increment))))
        return round(math.floor(qty / increment) * increment, decimals)

    def calculate_position_size(
        self,
        signal: "TradeSignal",
        balance: float,
        risk_pct: float,
        entry_price: Optional[float] = None,
    ) -> float:
        """
        Calculate position size with the same risk model for both modes.
        risk_amount = balance * risk_pct
        position_size = risk_amount / abs(entry - stop)
        """
        if signal.pair != XRP_ONLY_SYMBOL:
            raise RuntimeError(f"XRP-only mode: unsupported signal pair {signal.pair}")
        risk_amount = balance * risk_pct
        effective_entry = float(entry_price if entry_price is not None else signal.entry_price)
        stop_distance = abs(effective_entry - signal.stop_price)
        if stop_distance <= 0:
            logger.error("Stop distance is zero, cannot size position")
            return 0.0

        raw_qty = risk_amount / stop_distance

        # Keep sizing consistent with deterministic validator caps.
        max_position_value = float(balance) * 0.2
        max_qty_by_position = max_position_value / float(effective_entry) if effective_entry > 0 else 0.0
        max_qty_by_risk = risk_amount / float(effective_entry) if effective_entry > 0 else 0.0

        limits = load_limits()
        cap_candidates = [raw_qty, max_qty_by_position, max_qty_by_risk]
        if effective_entry > 0 and limits.max_position_pct > 0:
            cap_candidates.append((float(balance) * float(limits.max_position_pct)) / float(effective_entry))
        if effective_entry > 0 and limits.max_notional_per_trade > 0:
            cap_candidates.append(float(limits.max_notional_per_trade) / float(effective_entry))

        capped_qty = min(cap_candidates)
        qty = self.round_quantity(capped_qty, signal.pair)

        info = self.get_symbol_info(signal.pair)
        if qty < info["base_min_size"]:
            logger.warning(f"Position size {qty} below minimum {info['base_min_size']}")
            return 0.0

        cost = qty * effective_entry
        if cost > balance * 0.95:
            qty = self.round_quantity((balance * 0.90) / effective_entry, signal.pair)

        return qty

    def execute_signal(
        self,
        signal: "TradeSignal",
        balance: Optional[float] = None,
        risk_multiplier: float = 1.0,
    ) -> Optional[str]:
        if signal.pair != XRP_ONLY_SYMBOL:
            raise RuntimeError(f"XRP-only mode: unsupported signal pair {signal.pair}")
        if is_safe_mode():
            logger.bind(event="TRADE_BLOCKED", pair=signal.pair, reason="safe_mode").warning(
                "TRADE_BLOCKED"
            )
            return None

        trade_id = f"{self.mode.lower()}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        available_balance = self.get_balance() if balance is None else balance

        side = str(getattr(signal, "side", "BUY")).upper()
        bounded_multiplier = max(0.1, min(2.0, float(risk_multiplier)))
        risk_pct = signal.confidence * float(getattr(signal, "risk_per_trade", settings.trading.risk_per_trade)) * bounded_multiplier

        try:
            ticker = self.fetcher.get_ticker(signal.pair)
        except Exception as exc:
            logger.error(f"[VALIDATION] Unable to fetch ticker for {signal.pair}: {exc}")
            return None

        mark_price = float(ticker.get("price") or 0.0)
        market_entry_price = mark_price

        model_entry_price = float(getattr(signal, "model_entry_price", signal.entry_price) or signal.entry_price or 0.0)
        if mark_price <= 0 or market_entry_price <= 0:
            logger.warning(
                f"[VALIDATION] Trade blocked for {signal.pair}: invalid market price mark={mark_price:.8f} entry={market_entry_price:.8f}"
            )
            return None

        model_vs_mark_pct = abs(model_entry_price - mark_price) / mark_price
        entry_vs_mark_pct = abs(market_entry_price - mark_price) / mark_price
        logger.bind(
            event="PRICE_DIAGNOSTIC",
            pair=signal.pair,
            side=side,
            raw_mark_price=mark_price,
            model_entry_price=model_entry_price,
            market_entry_price=market_entry_price,
            model_vs_mark_pct=model_vs_mark_pct,
            entry_vs_mark_pct=entry_vs_mark_pct,
        ).info("PRICE_DIAGNOSTIC")

        if model_vs_mark_pct > 0.05:
            logger.bind(
                event="PRICE_DESYNC_ERROR",
                pair=signal.pair,
                side=side,
                raw_mark_price=mark_price,
                model_entry_price=model_entry_price,
                difference_pct=model_vs_mark_pct,
                threshold=0.05,
            ).error("PRICE_DESYNC_ERROR")
            return None

        if entry_vs_mark_pct >= 0.02:
            logger.bind(
                event="PRICE_INVARIANT_REJECTED",
                pair=signal.pair,
                side=side,
                raw_mark_price=mark_price,
                market_entry_price=market_entry_price,
                difference_pct=entry_vs_mark_pct,
                threshold=0.02,
            ).error("PRICE_INVARIANT_REJECTED")
            return None
        assert abs(market_entry_price - mark_price) / mark_price < 0.02

        base_quantity = self.calculate_position_size(
            signal,
            available_balance,
            risk_pct,
            entry_price=market_entry_price,
        )
        size_multiplier = max(0.1, min(3.0, float(getattr(signal, "position_size_multiplier", 1.0))))
        quantity = self.round_quantity(base_quantity * size_multiplier, signal.pair)
        if quantity <= 0:
            logger.warning(f"Cannot size position for {signal.pair}")
            return None

        notional = float(market_entry_price) * float(quantity)
        if abnormal_trade_detector(signal.pair, float(market_entry_price), float(quantity), float(available_balance)):
            logger.bind(
                event="TRADE_BLOCKED",
                pair=signal.pair,
                price=float(market_entry_price),
                qty=float(quantity),
                balance=float(available_balance),
                notional=notional,
                reason="abnormal_trade_detector",
            ).warning("TRADE_BLOCKED")
            return None

        if not validate_position_size(signal.pair, notional, float(available_balance)):
            return None

        trade_data = {
            "trade_id": trade_id,
            "symbol": signal.pair,
            "entry_price": market_entry_price,
            "stop_price": signal.stop_price,
            "target_price": signal.target_price,
            "confidence": float(signal.confidence),
            "quantity": quantity,
            "mark_price": mark_price,
            "risk_per_trade": risk_pct,
            "balance": available_balance,
        }

        validation = validate_trade(trade_data, available_balance)
        if not validation.get("valid", False):
            reason = validation.get("reason", "Unknown validation failure")
            logger.warning(f"[VALIDATION] Trade blocked for {signal.pair}: {reason}")
            return None

        if AUDITOR_READY and settings.runtime.ai_auditor_enabled:
            audit_result = audit_trade(trade_data)
        else:
            audit_result = {"status": "skipped"}

        if not audit_result.get("valid", True):
            logger.warning(
                f"[AI_AUDIT] Trade blocked for {signal.pair}: {audit_result.get('reason', 'No reason')}"
            )
            return None

        logger.info(
            f"[{self.mode}] Executing {signal.pair}: qty={quantity}, "
            f"entry~{market_entry_price:.6f}, risk_multiplier={bounded_multiplier:.2f}"
        )

        try:
            if side == "SELL":
                entry_result = self.sell(signal.pair, quantity, price=market_entry_price)
                stop_side = "buy"
                target_side = "buy"
            else:
                entry_result = self.buy(
                    signal.pair,
                    quantity,
                    price=market_entry_price,
                    market_ticker=ticker,
                )
                stop_side = "sell"
                target_side = "sell"

            fill_price = float(entry_result.get("fill_price", market_entry_price))
            executed_qty = float(entry_result.get("quantity", quantity))
            entry_fee = float(entry_result.get("fee", 0.0))
            latency_ms = float(entry_result.get("latency_ms", 0.0))
            slippage_bps = (
                abs(fill_price - market_entry_price) / market_entry_price * 10000
                if market_entry_price > 0
                else 0.0
            )

            self.place_stop_order(signal.pair, stop_side, executed_qty, signal.stop_price)
            self.place_limit_order(signal.pair, target_side, executed_qty, signal.target_price)

            self._record_open_trade(
                trade_id=trade_id,
                signal=signal,
                quantity=executed_qty,
                fill_price=fill_price,
                commission=entry_fee,
                slippage_bps=slippage_bps,
                latency_ms=latency_ms,
            )

            log_trade(
                {
                    "event": "entry",
                    "trade_id": trade_id,
                    "pair": signal.pair,
                    "entry_price": fill_price,
                    "side": side,
                    "stop_price": signal.stop_price,
                    "target_price": signal.target_price,
                    "quantity": executed_qty,
                    "confidence": signal.confidence,
                    "slippage_bps": slippage_bps,
                    "commission": entry_fee,
                    "paper": self.mode == "PAPER",
                    "mode": self.mode,
                }
            )

            logger.info(
                f"[{self.mode}] Trade opened: {trade_id} | {signal.pair} | "
                f"Entry: {fill_price:.6f} | Stop: {signal.stop_price:.6f} | "
                f"Target: {signal.target_price:.6f}"
            )
            return trade_id
        except Exception as exc:
            logger.error(f"Execution failed for {signal.pair}: {exc}")
            self.on_execution_failed(signal.pair, quantity)
            return None

    def _record_open_trade(
        self,
        trade_id: str,
        signal: "TradeSignal",
        quantity: float,
        fill_price: float,
        commission: float,
        slippage_bps: float,
        latency_ms: float,
    ) -> None:
        if fill_price <= 0:
            logger.error(
                "[VALIDATION] Trade {trade_id} has non-positive entry price {fill_price:.8f}, skipping DB record",
                trade_id=trade_id,
                fill_price=fill_price,
            )
            return
        session = get_session()
        try:
            trade = Trade(
                trade_id=trade_id,
                pair=signal.pair,
                side=str(getattr(signal, "side", "BUY")).upper(),
                entry_time=datetime.utcnow(),
                entry_price=fill_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                quantity=quantity,
                status="open",
                confidence=signal.confidence,
                ai_score=signal.ai_score,
                base_ai_score=signal.base_ai_score,
                tft_score=float(getattr(signal, "tft_score", 0.0)),
                xgb_score=float(getattr(signal, "xgb_score", 0.0)),
                ppo_score=float(getattr(signal, "ppo_score", 0.0)),
                gov_adjust=float(getattr(signal, "gov_adjust", 0.0)),
                final_ai_score=float(getattr(signal, "final_ai_score", signal.ai_score)),
                weight_snapshot_json=json.dumps(getattr(signal, "weight_snapshot", {}), sort_keys=True),
                governance_code=signal.governance_code or None,
                model_version=self.fetcher.__class__.__name__,
                features_at_entry=signal.features_snapshot,
                prediction={
                    "prob_up": signal.prob_up,
                    "prob_down": signal.prob_down,
                    "expected_move": signal.expected_move,
                    "confidence": signal.confidence,
                    "forecast_vector": signal.forecast_vector,
                    "ai_score": signal.ai_score,
                    "base_ai_score": signal.base_ai_score,
                    "ai_confidence": signal.ai_confidence,
                    "meta_probability": signal.meta_probability,
                    "meta_model_version": signal.meta_model_version,
                    "tft_model_version": signal.tft_model_version,
                    "governance_code": signal.governance_code,
                    "governance_approved": signal.governance_approved,
                    "governance_size_mult": signal.governance_size_mult,
                    "governance_conf_adj": signal.governance_conf_adj,
                    "governance_risk_mode": signal.governance_risk_mode,
                    "tft_score": float(getattr(signal, "tft_score", 0.0)),
                    "xgb_score": float(getattr(signal, "xgb_score", 0.0)),
                    "ppo_score": float(getattr(signal, "ppo_score", 0.0)),
                    "gov_adjust": float(getattr(signal, "gov_adjust", 0.0)),
                    "final_ai_score": float(getattr(signal, "final_ai_score", signal.ai_score)),
                    "weight_snapshot": getattr(signal, "weight_snapshot", {}),
                    "risk_per_trade": float(getattr(signal, "risk_per_trade", settings.trading.risk_per_trade)),
                    "side": str(getattr(signal, "side", "BUY")).upper(),
                    "position_size_multiplier": float(getattr(signal, "position_size_multiplier", 1.0)),
                    "adaptive_threshold": float(getattr(signal, "adaptive_threshold", 0.0)),
                    "regime": getattr(signal, "regime", {}),
                },
                ai_reasoning=signal.reasoning,
                slippage_bps=slippage_bps,
                commission=commission,
                latency_ms=latency_ms,
            )
            session.add(trade)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Error recording trade: {exc}")
        finally:
            session.close()
