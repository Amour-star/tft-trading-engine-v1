"""
Live exchange executor for real KuCoin order placement.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from data.database import Trade, get_session
from utils.logging import log_api_error

from execution.base_executor import BaseExecutor


class LiveExecutor(BaseExecutor):
    """Executor that routes orders to the real exchange."""

    mode = "LIVE"

    def get_balance(self) -> float:
        return self.fetcher.get_balance("USDT")

    def get_positions(self) -> Dict[str, Any]:
        return self.fetcher.get_all_balances()

    def buy(self, symbol: str, qty: float, price: Optional[float] = None) -> Dict[str, Any]:
        _ = price
        result = self.place_market_buy(symbol, qty)
        fill = self._wait_for_fill(result["order_id"], timeout=10)
        if fill is None:
            raise RuntimeError(f"Buy order not filled: {result['order_id']}")
        result.update(fill)
        return result

    def sell(self, symbol: str, qty: float, price: Optional[float] = None) -> Dict[str, Any]:
        _ = price
        result = self.market_close(symbol, qty)
        fill = self._wait_for_fill(result["order_id"], timeout=10)
        if fill:
            result.update(fill)
        return result

    def close_position(self, symbol: str) -> Dict[str, Any]:
        balances = self.fetcher.get_all_balances()
        base = symbol.split("-")[0]
        qty = float(balances.get(base, 0.0))
        if qty <= 0:
            return {"status": "no_position", "symbol": symbol, "quantity": 0.0}
        return self.sell(symbol, qty)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def place_market_buy(self, symbol: str, quantity: float) -> Dict[str, Any]:
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
            logger.info(
                f"Market BUY placed: {symbol} qty={quantity} order={order_id} latency={latency:.0f}ms"
            )

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
            logger.info(f"Stop order placed: {symbol} {side} @ {stop_price} order={order_id}")
            return {"order_id": order_id, "stop_price": stop_price, "status": "placed"}
        except Exception as exc:
            log_api_error("create_stop_order", str(exc), symbol=symbol)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def place_limit_sell(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def cancel_order(self, order_id: str) -> bool:
        try:
            self.fetcher.trade_client.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as exc:
            log_api_error("cancel_order", str(exc), order_id=order_id)
            return False

    def cancel_all_orders(self, symbol: str) -> None:
        try:
            orders = self.fetcher.trade_client.get_order_list(symbol=symbol, status="active")
            for order in orders.get("items", []):
                self.cancel_order(order["id"])
        except Exception as exc:
            logger.warning(f"Error cancelling orders for {symbol}: {exc}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def market_close(self, symbol: str, quantity: float) -> Dict[str, Any]:
        client_oid = str(uuid.uuid4())[:32]
        try:
            result = self.fetcher.trade_client.create_market_order(
                symbol=symbol,
                side="sell",
                size=str(quantity),
                client_oid=client_oid,
            )
            order_id = result.get("orderId", "")
            logger.info(f"Market CLOSE: {symbol} qty={quantity} order={order_id}")
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
        _ = quantity
        try:
            self.cancel_all_orders(symbol)
            balances = self.fetcher.get_all_balances()
            base = symbol.split("-")[0]
            if base in balances and balances[base] > 0:
                self.market_close(symbol, balances[base])
                logger.warning(f"Emergency cleanup: sold {balances[base]} {base}")
        except Exception as exc:
            logger.critical(f"Emergency cleanup failed for {symbol}: {exc}")

    def reconcile_positions(self) -> None:
        """
        On restart, reconcile database state with exchange state.
        Ensures no orphaned positions.
        """
        logger.info("Reconciling positions...")
        session = get_session()
        try:
            open_trades = session.query(Trade).filter(Trade.status == "open").all()
            balances = self.fetcher.get_all_balances()

            for trade in open_trades:
                base = trade.pair.split("-")[0]
                if base not in balances or balances[base] < trade.quantity * 0.9:
                    logger.warning(
                        f"Orphaned trade {trade.trade_id}: position not found on exchange"
                    )
                    trade.status = "closed"
                    trade.exit_reason = "reconciliation"
                    trade.exit_time = datetime.utcnow()
                    try:
                        ticker = self.fetcher.get_ticker(trade.pair)
                        trade.exit_price = ticker["price"]
                        if trade.entry_price:
                            trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                            trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
                    except Exception:
                        pass
                else:
                    logger.info(f"Trade {trade.trade_id} confirmed on exchange")

            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Reconciliation error: {exc}")
        finally:
            session.close()
