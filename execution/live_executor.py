"""
Live exchange executor for real KuCoin order placement.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import XRP_ONLY_SYMBOL
from data.database import Trade, get_session
from risk.safety_layer import validate_price
from utils.logging import log_api_error

from execution.base_executor import BaseExecutor


class LiveExecutor(BaseExecutor):
    """Executor that routes orders to the real exchange."""

    mode = "LIVE"

    def __init__(self, fetcher) -> None:
        super().__init__(fetcher)
        self._validate_live_clients()
        self._last_prices: Dict[str, float] = {}

    def _validate_live_clients(self) -> None:
        if self.fetcher.trade_client is None or self.fetcher.user_client is None:
            raise RuntimeError(
                "LIVE mode requires valid KuCoin API credentials and initialized API clients."
            )

    @staticmethod
    def _assert_symbol(symbol: str) -> None:
        if symbol != XRP_ONLY_SYMBOL:
            raise RuntimeError(f"XRP-only mode: unsupported symbol {symbol}")

    def get_balance(self) -> float:
        self._validate_live_clients()
        return self.fetcher.get_balance("USDT")

    def get_positions(self) -> Dict[str, Any]:
        self._validate_live_clients()
        return self.fetcher.get_all_balances()

    def buy(
        self,
        symbol: str,
        qty: float,
        price: Optional[float] = None,
        market_ticker: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._assert_symbol(symbol)
        _ = price
        try:
            ticker = market_ticker or self.fetcher.get_ticker(symbol)
            mark_price = float(ticker.get("price") or 0.0)
        except Exception as exc:
            logger.bind(event="VALIDATION_ERROR", pair=symbol, error=str(exc)).error(
                "VALIDATION_ERROR"
            )
            raise

        last_price = self._last_prices.get(symbol)
        if not validate_price(symbol, mark_price, last_price):
            raise ValueError("Invalid market price")
        self._last_prices[symbol] = float(mark_price)

        result = self.place_market_buy(symbol, qty)
        fill = self._wait_for_fill(result["order_id"], timeout=10)
        if fill is None:
            raise RuntimeError(f"Buy order not filled: {result['order_id']}")
        result.update(fill)
        return result

    def sell(self, symbol: str, qty: float, price: Optional[float] = None) -> Dict[str, Any]:
        self._assert_symbol(symbol)
        _ = price
        try:
            ticker = self.fetcher.get_ticker(symbol)
            mark_price = float(ticker.get("price") or 0.0)
        except Exception as exc:
            logger.bind(event="VALIDATION_ERROR", pair=symbol, error=str(exc)).error(
                "VALIDATION_ERROR"
            )
            raise

        last_price = self._last_prices.get(symbol)
        if not validate_price(symbol, mark_price, last_price):
            raise ValueError("Invalid market price")
        self._last_prices[symbol] = float(mark_price)

        result = self.market_close(symbol, qty)
        fill = self._wait_for_fill(result["order_id"], timeout=10)
        if fill:
            result.update(fill)
        return result

    def close_position(self, symbol: str) -> Dict[str, Any]:
        self._assert_symbol(symbol)
        balances = self.fetcher.get_all_balances()
        base = symbol.split("-")[0]
        qty = float(balances.get(base, 0.0))
        if qty <= 0:
            return {"status": "no_position", "symbol": symbol, "quantity": 0.0}
        return self.sell(symbol, qty)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def place_market_buy(self, symbol: str, quantity: float) -> Dict[str, Any]:
        self._assert_symbol(symbol)
        client_oid = str(uuid.uuid4())[:32]
        start_time = time.time()

        try:
            result = self.fetcher.trade_client.create_market_order(
                symbol=symbol,
                side="buy",
                size=str(quantity),
                client_oid=client_oid,
            )
            latency = (time.time() - start_time) * 1000
            order_id = result.get("orderId", "")
            logger.bind(
                event="ORDER_PLACED",
                pair=symbol,
                side="buy",
                qty=float(quantity),
                order_id=order_id,
                latency_ms=float(latency),
            ).info("ORDER_PLACED")

            return {
                "order_id": order_id,
                "client_oid": client_oid,
                "symbol": symbol,
                "side": "buy",
                "quantity": quantity,
                "latency_ms": latency,
                "status": "placed",
            }
        except Exception as exc:
            log_api_error("create_market_order", str(exc), symbol=symbol)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def place_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
    ) -> Dict[str, Any]:
        self._assert_symbol(symbol)
        stop_price = self.round_price(stop_price, symbol)
        client_oid = str(uuid.uuid4())[:32]

        try:
            result = self.fetcher.trade_client.create_market_order(
                symbol=symbol,
                side=side,
                size=str(quantity),
                client_oid=client_oid,
                stop="loss",
                stop_price=str(stop_price),
            )
            order_id = result.get("orderId", "")
            logger.bind(
                event="ORDER_PLACED",
                pair=symbol,
                side=side,
                qty=float(quantity),
                stop_price=float(stop_price),
                order_id=order_id,
                order_type="stop",
            ).info("ORDER_PLACED")
            return {"order_id": order_id, "stop_price": stop_price, "status": "placed"}
        except Exception as exc:
            log_api_error("create_stop_order", str(exc), symbol=symbol)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def place_limit_sell(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        self._assert_symbol(symbol)
        price = self.round_price(price, symbol)
        quantity = self.round_quantity(quantity, symbol)
        client_oid = str(uuid.uuid4())[:32]

        try:
            result = self.fetcher.trade_client.create_limit_order(
                symbol=symbol,
                side="sell",
                price=str(price),
                size=str(quantity),
                client_oid=client_oid,
            )
            order_id = result.get("orderId", "")
            logger.info(f"Limit SELL placed: {symbol} @ {price} qty={quantity} order={order_id}")
            return {"order_id": order_id, "price": price, "status": "placed"}
        except Exception as exc:
            log_api_error("create_limit_order", str(exc), symbol=symbol)
            raise

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> Dict[str, Any]:
        self._assert_symbol(symbol)
        side_l = str(side).lower()
        if side_l == "sell":
            return self.place_limit_sell(symbol, quantity, price)

        price = self.round_price(price, symbol)
        quantity = self.round_quantity(quantity, symbol)
        client_oid = str(uuid.uuid4())[:32]
        try:
            result = self.fetcher.trade_client.create_limit_order(
                symbol=symbol,
                side=side_l,
                price=str(price),
                size=str(quantity),
                client_oid=client_oid,
            )
            order_id = result.get("orderId", "")
            logger.info(f"Limit {side_l.upper()} placed: {symbol} @ {price} qty={quantity} order={order_id}")
            return {"order_id": order_id, "price": price, "status": "placed", "side": side_l}
        except Exception as exc:
            log_api_error("create_limit_order", str(exc), symbol=symbol)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def cancel_order(self, order_id: str) -> bool:
        try:
            self.fetcher.trade_client.cancel_order(order_id)
            logger.bind(event="ORDER_CANCELLED", order_id=order_id).info("ORDER_CANCELLED")
            return True
        except Exception as exc:
            log_api_error("cancel_order", str(exc), order_id=order_id)
            return False

    def cancel_all_orders(self, symbol: str) -> None:
        self._assert_symbol(symbol)
        try:
            orders = self.fetcher.trade_client.get_order_list(symbol=symbol, status="active")
            for order in orders.get("items", []):
                self.cancel_order(order["id"])
        except Exception as exc:
            logger.bind(event="ORDER_CANCEL_ALL_FAILED", pair=symbol, error=str(exc)).warning(
                "ORDER_CANCEL_ALL_FAILED"
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def market_close(self, symbol: str, quantity: float) -> Dict[str, Any]:
        self._assert_symbol(symbol)
        client_oid = str(uuid.uuid4())[:32]
        try:
            result = self.fetcher.trade_client.create_market_order(
                symbol=symbol,
                side="sell",
                size=str(quantity),
                client_oid=client_oid,
            )
            order_id = result.get("orderId", "")
            logger.bind(
                event="ORDER_PLACED",
                pair=symbol,
                side="sell",
                qty=float(quantity),
                order_id=order_id,
                order_type="close",
            ).info("ORDER_PLACED")
            return {"order_id": order_id, "status": "closed", "symbol": symbol, "quantity": quantity}
        except Exception as exc:
            log_api_error("market_close", str(exc), symbol=symbol)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def get_order_detail(self, order_id: str) -> Dict[str, Any]:
        try:
            detail = self.fetcher.trade_client.get_order_details(order_id)
            return {
                "order_id": detail.get("id", ""),
                "symbol": detail.get("symbol", ""),
                "side": detail.get("side", ""),
                "type": detail.get("type", ""),
                "price": float(detail.get("price", 0)),
                "size": float(detail.get("size", 0)),
                "deal_size": float(detail.get("dealSize", 0)),
                "deal_funds": float(detail.get("dealFunds", 0)),
                "fee": float(detail.get("fee", 0)),
                "is_active": detail.get("isActive", False),
                "cancel_exist": detail.get("cancelExist", False),
            }
        except Exception as exc:
            log_api_error("get_order_detail", str(exc), order_id=order_id)
            raise

    def _wait_for_fill(self, order_id: str, timeout: int = 10) -> Optional[Dict[str, float]]:
        start = time.time()
        while time.time() - start < timeout:
            detail = self.get_order_detail(order_id)
            deal_size = detail.get("deal_size", 0)
            if deal_size > 0:
                fill_price = detail["deal_funds"] / deal_size
                return {
                    "fill_price": fill_price,
                    "fee": float(detail.get("fee", 0.0)),
                }
            time.sleep(0.5)
        return None

    def on_execution_failed(self, symbol: str, quantity: float) -> None:
        self._emergency_cleanup(symbol, quantity)

    def _emergency_cleanup(self, symbol: str, quantity: float) -> None:
        self._assert_symbol(symbol)
        _ = quantity
        try:
            self.cancel_all_orders(symbol)
            balances = self.fetcher.get_all_balances()
            base = symbol.split("-")[0]
            if base in balances and balances[base] > 0:
                self.market_close(symbol, balances[base])
                logger.bind(
                    event="EMERGENCY_CLEANUP",
                    pair=symbol,
                    base=base,
                    qty=float(balances[base]),
                ).warning("EMERGENCY_CLEANUP")
        except Exception as exc:
            logger.bind(event="EMERGENCY_CLEANUP_FAILED", pair=symbol, error=str(exc)).critical(
                "EMERGENCY_CLEANUP_FAILED"
            )

    def reconcile_positions(self) -> None:
        symbols = {XRP_ONLY_SYMBOL}
        session = get_session()
        try:
            open_pairs = (
                session.query(Trade.pair)
                .filter(Trade.status == "open")
                .all()
            )
            for (pair,) in open_pairs:
                if pair:
                    symbols.add(str(pair))
        finally:
            session.close()

        events = self.schedule_reconciliation(sorted(symbols), source="live_executor")
        logger.bind(
            reconciliation={
                "symbol": ",".join(sorted(symbols)),
                "event": "reconcile_request",
                "details": {"queued_events": len(events), "mode": "LIVE"},
            }
        ).info("RECONCILIATION_EVENT")
