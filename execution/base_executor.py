"""
Abstract execution interface and shared trade pipeline.
"""
from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger

from config.settings import settings
from data.database import Trade, get_session
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
    def buy(self, symbol: str, qty: float, price: Optional[float] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def sell(self, symbol: str, qty: float, price: Optional[float] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def close_position(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError

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

    def cancel_order(self, order_id: str) -> bool:
        _ = order_id
        return False

    def cancel_all_orders(self, symbol: str) -> None:
        _ = symbol

    def reconcile_positions(self) -> None:
        """Optional startup reconciliation hook."""

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
    ) -> float:
        """
        Calculate position size with the same risk model for both modes.
        risk_amount = balance * risk_pct
        position_size = risk_amount / abs(entry - stop)
        """
        risk_amount = balance * risk_pct
        stop_distance = abs(signal.entry_price - signal.stop_price)
        if stop_distance <= 0:
            logger.error("Stop distance is zero, cannot size position")
            return 0.0

        raw_qty = risk_amount / stop_distance
        qty = self.round_quantity(raw_qty, signal.pair)

        info = self.get_symbol_info(signal.pair)
        if qty < info["base_min_size"]:
            logger.warning(f"Position size {qty} below minimum {info['base_min_size']}")
            return 0.0

        cost = qty * signal.entry_price
        if cost > balance * 0.95:
            qty = self.round_quantity((balance * 0.90) / signal.entry_price, signal.pair)

        return qty

    def execute_signal(self, signal: "TradeSignal", balance: Optional[float] = None) -> Optional[str]:
        trade_id = f"{self.mode.lower()}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        available_balance = self.get_balance() if balance is None else balance

        risk_pct = signal.confidence * settings.trading.risk_per_trade
        quantity = self.calculate_position_size(signal, available_balance, risk_pct)
        if quantity <= 0:
            logger.warning(f"Cannot size position for {signal.pair}")
            return None

        logger.info(
            f"[{self.mode}] Executing {signal.pair}: qty={quantity}, entry~{signal.entry_price:.6f}"
        )

        try:
            buy_result = self.buy(signal.pair, quantity, price=signal.entry_price)
            fill_price = float(buy_result.get("fill_price", signal.entry_price))
            entry_fee = float(buy_result.get("fee", 0.0))
            latency_ms = float(buy_result.get("latency_ms", 0.0))
            slippage_bps = (
                abs(fill_price - signal.entry_price) / signal.entry_price * 10000
                if signal.entry_price > 0
                else 0.0
            )

            self.place_stop_order(signal.pair, "sell", quantity, signal.stop_price)
            self.place_limit_sell(signal.pair, quantity, signal.target_price)

            self._record_open_trade(
                trade_id=trade_id,
                signal=signal,
                quantity=quantity,
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
                    "stop_price": signal.stop_price,
                    "target_price": signal.target_price,
                    "quantity": quantity,
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
        session = get_session()
        try:
            trade = Trade(
                trade_id=trade_id,
                pair=signal.pair,
                side="BUY",
                entry_time=datetime.utcnow(),
                entry_price=fill_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                quantity=quantity,
                status="open",
                confidence=signal.confidence,
                model_version=self.fetcher.__class__.__name__,
                features_at_entry=signal.features_snapshot,
                prediction={
                    "prob_up": signal.prob_up,
                    "prob_down": signal.prob_down,
                    "expected_move": signal.expected_move,
                    "confidence": signal.confidence,
                    "forecast_vector": signal.forecast_vector,
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
