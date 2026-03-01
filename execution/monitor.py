"""
Position monitor: real-time trade management.
Monitors open positions and adjusts stops/exits based on market conditions.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from data.database import Trade, get_session
from data.fetcher import KuCoinDataFetcher
from data.features import compute_features
from execution.base_executor import BaseExecutor
from models.tft_model import TFTPredictor
from utils.logging import log_trade
from utils.pnl import calculate_realized_pnl


class PositionMonitor:
    """
    Monitors open positions in real time.
    Implements trailing stops, confidence-based exits, and PPO-based management.
    """

    def __init__(
        self,
        fetcher: KuCoinDataFetcher,
        executor: BaseExecutor,
        predictor: TFTPredictor,
        rl_manager: Optional[Any] = None,
    ) -> None:
        self.fetcher = fetcher
        self.executor = executor
        self.predictor = predictor
        self.rl_manager = rl_manager
        self._trailing_extreme: Dict[str, float] = {}

    def monitor_cycle(self) -> Optional[str]:
        """
        Run one monitoring cycle for open position.
        Returns trade_id if position was closed, None otherwise.
        """
        session = get_session()
        try:
            trade = session.query(Trade).filter(Trade.status == "open").first()
            if not trade:
                return None

            pair = trade.pair
            side = str(trade.side or "BUY").upper()
            side_mult = 1.0 if side == "BUY" else -1.0
            entry_price = float(trade.entry_price)
            stop_price = float(trade.stop_price)
            target_price = float(trade.target_price)

            try:
                ticker = self.fetcher.get_ticker(pair)
                current_price = float(ticker["price"])
            except Exception as e:
                logger.error(f"Cannot get price for {pair}: {e}")
                return None

            extreme = self._trailing_extreme.get(trade.trade_id, entry_price)
            if side == "BUY":
                extreme = max(extreme, current_price)
            else:
                extreme = min(extreme, current_price)
            self._trailing_extreme[trade.trade_id] = extreme

            latest_features, prediction, atr = self._build_model_context(pair)

            # Check 1: stop/target hard exits.
            if (side == "BUY" and current_price <= stop_price) or (side == "SELL" and current_price >= stop_price):
                return self._close_trade(trade, current_price, "stop", session)
            if (side == "BUY" and current_price >= target_price) or (side == "SELL" and current_price <= target_price):
                return self._close_trade(trade, current_price, "target", session)

            # Check 2: trailing stop when profitable.
            pnl_pct = side_mult * (current_price - entry_price) / max(entry_price, 1e-12)
            if pnl_pct > 0:
                if pnl_pct > 0.02:
                    if side == "BUY":
                        trail_stop = extreme - (atr * 1.5)
                        should_update = trail_stop > stop_price
                    else:
                        trail_stop = extreme + (atr * 1.5)
                        should_update = trail_stop < stop_price
                    if should_update:
                        logger.info(f"Trailing stop: {stop_price:.6f} -> {trail_stop:.6f}")
                        trade.stop_price = trail_stop
                        stop_price = trail_stop
                        self._update_stop_order(trade, trail_stop)

            # Check 3: PPO action for position management.
            if self.rl_manager is not None:
                rl_closed_id = self._apply_rl_management(
                    trade=trade,
                    current_price=current_price,
                    stop_price=stop_price,
                    atr=atr,
                    latest_features=latest_features,
                    session=session,
                )
                if rl_closed_id:
                    return rl_closed_id

            # Check 4: model confidence reversal guard.
            new_confidence = float(prediction.get("confidence", 0.0))
            prob_down = float(prediction.get("prob_down", 0.0))
            prob_up = float(prediction.get("prob_up", 0.0))
            if new_confidence < 0.4:
                if side == "BUY":
                    tighter_stop = current_price - (current_price - stop_price) * 0.5
                    should_update = tighter_stop > stop_price
                else:
                    tighter_stop = current_price + (stop_price - current_price) * 0.5
                    should_update = tighter_stop < stop_price
                if should_update:
                    logger.info(f"Tightening stop on low confidence: {new_confidence:.3f}")
                    trade.stop_price = tighter_stop
                    self._update_stop_order(trade, tighter_stop)
            if side == "BUY" and prob_down > 0.75 and new_confidence > 0.6:
                logger.info(f"Strong bearish signal: prob_down={prob_down:.3f}")
                return self._close_trade(trade, current_price, "signal_reversal", session)
            if side == "SELL" and prob_up > 0.75 and new_confidence > 0.6:
                logger.info(f"Strong bullish signal against short: prob_up={prob_up:.3f}")
                return self._close_trade(trade, current_price, "signal_reversal", session)

            # Check 5: short-term momentum weakening.
            try:
                df5 = self.fetcher.fetch_klines(pair, "5min")
                if not df5.empty and len(df5) > 20:
                    recent_returns = df5["close"].pct_change().tail(5)
                    if side == "BUY":
                        weakening = all(r < 0 for r in recent_returns.dropna())
                    else:
                        weakening = all(r > 0 for r in recent_returns.dropna())
                    if weakening:
                        logger.info("Momentum weakening detected")
                        if pnl_pct > 0.005:
                            return self._close_trade(trade, current_price, "momentum_weakening", session)
            except Exception:
                pass

            session.commit()
            return None

        except Exception as e:
            logger.error(f"Monitor cycle error: {e}")
            session.rollback()
            return None
        finally:
            session.close()

    def _build_model_context(self, pair: str) -> tuple[Dict[str, Any], Dict[str, Any], float]:
        latest_features: Dict[str, Any] = {}
        prediction: Dict[str, Any] = {}
        atr = self._get_current_atr(pair)

        try:
            df = self.fetcher.fetch_klines(pair, "15min")
            if not df.empty and len(df) > 100:
                df["pair"] = pair
                df = compute_features(df, None)
                latest_features = df.iloc[-1].to_dict()
                prediction = self.predictor.predict(df, pair)
                atr = float(latest_features.get("atr_14", atr))
        except Exception as e:
            logger.warning(f"Model context failed for {pair}: {e}")

        return latest_features, prediction, atr

    def _apply_rl_management(
        self,
        trade: Trade,
        current_price: float,
        stop_price: float,
        atr: float,
        latest_features: Dict[str, Any],
        session,
    ) -> Optional[str]:
        try:
            quantity = float(trade.quantity or 0.0)
            if quantity <= 0:
                return self._close_trade(trade, current_price, "rl_zero_quantity", session)
            side = str(trade.side or "BUY").upper()
            side_mult = 1.0 if side == "BUY" else -1.0

            elapsed_min = max(0.0, (datetime.utcnow() - trade.entry_time).total_seconds() / 60.0)
            unrealized_pnl = side_mult * (current_price - float(trade.entry_price)) * quantity
            rl_state = {
                "entry_price": float(trade.entry_price),
                "current_price": current_price,
                "quantity": quantity,
                "unrealized_pnl": unrealized_pnl,
                "time_in_trade": elapsed_min,
                "rsi": float(latest_features.get("rsi_14", 50.0)),
                "ema_20": float(latest_features.get("ema_21", current_price)),
                "volatility": float(latest_features.get("volatility_20", 0.0)),
            }
            rl_action = self.rl_manager.step(rl_state)

            payload = dict(trade.prediction or {})
            payload["rl_last_action"] = rl_action.action
            payload["rl_model_version"] = rl_action.model_version
            trade.prediction = payload

            if rl_action.should_exit:
                logger.info(f"RL exit for trade {trade.trade_id}")
                return self._close_trade(trade, current_price, "rl_exit", session)

            if atr > 0:
                if side == "BUY":
                    suggested_stop = current_price - atr * float(rl_action.stop_atr_multiplier)
                    should_update = suggested_stop > stop_price and suggested_stop < current_price
                else:
                    suggested_stop = current_price + atr * float(rl_action.stop_atr_multiplier)
                    should_update = suggested_stop < stop_price and suggested_stop > current_price
                if should_update:
                    trade.stop_price = suggested_stop
                    self._update_stop_order(trade, suggested_stop)

            min_qty = self._min_qty(trade.pair)

            if rl_action.action == "decrease" and quantity > min_qty:
                reduce_qty = quantity * 0.25
                if reduce_qty > min_qty and (quantity - reduce_qty) > min_qty:
                    close_res = self.executor.force_close(
                        trade.pair,
                        reduce_qty,
                        price=current_price,
                        side=side,
                    )
                    realized = float(close_res.get("realized_pnl", 0.0))
                    trade.quantity = quantity - reduce_qty
                    self._add_partial_realized_pnl(trade, realized)
                    logger.info(
                        f"RL decreased position {trade.trade_id}: -{reduce_qty:.8f}, "
                        f"remaining={trade.quantity:.8f}"
                    )
                    self._update_stop_order(trade, float(trade.stop_price))

            elif rl_action.action == "increase" and quantity > 0:
                add_qty = quantity * 0.15
                if add_qty > min_qty:
                    if side == "BUY":
                        add_res = self.executor.buy(trade.pair, add_qty, price=current_price)
                    else:
                        add_res = self.executor.sell(trade.pair, add_qty, price=current_price)
                    fill_price = float(add_res.get("fill_price", current_price))
                    fee = float(add_res.get("fee", 0.0))
                    new_qty = quantity + add_qty
                    trade.entry_price = (
                        (float(trade.entry_price) * quantity) + (fill_price * add_qty)
                    ) / new_qty
                    trade.quantity = new_qty
                    trade.commission = float(trade.commission or 0.0) + fee
                    logger.info(
                        f"RL increased position {trade.trade_id}: +{add_qty:.8f}, "
                        f"total={trade.quantity:.8f}"
                    )
                    self._update_stop_order(trade, float(trade.stop_price))

        except Exception as e:
            logger.warning(f"RL management failed for {trade.trade_id}: {e}")

        return None

    def _add_partial_realized_pnl(self, trade: Trade, realized: float) -> None:
        payload = dict(trade.prediction or {})
        payload["partial_realized_pnl"] = float(payload.get("partial_realized_pnl", 0.0)) + float(realized)
        trade.prediction = payload

    def _min_qty(self, symbol: str) -> float:
        try:
            info = self.executor.get_symbol_info(symbol)
            return float(info.get("base_min_size", 0.0))
        except Exception:
            return 0.0

    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        reason: str,
        session,
    ) -> str:
        """Close a trade and record the result."""
        trade_id = trade.trade_id
        pair = trade.pair
        side = str(trade.side or "BUY").upper()
        side_mult = 1.0 if side == "BUY" else -1.0
        quantity = float(trade.quantity)
        partial_realized = float((trade.prediction or {}).get("partial_realized_pnl", 0.0))

        try:
            self._cancel_all_orders(pair)
            close_result = self.executor.force_close(pair, quantity, price=exit_price, side=side)
            exit_price = float(close_result.get("fill_price", exit_price))
            exit_fee = float(close_result.get("fee", 0.0))
            realized_pnl = close_result.get("realized_pnl")
        except Exception as e:
            logger.error(f"Error closing position on exchange: {e}")
            exit_fee = 0.0
            realized_pnl = None

        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow()
        trade.exit_reason = reason
        trade.status = "closed"

        if trade.entry_price:
            trade.pnl = (
                float(realized_pnl) + partial_realized
                if realized_pnl is not None
                else calculate_realized_pnl(
                    entry_price=float(trade.entry_price),
                    exit_price=exit_price,
                    quantity=quantity,
                    side=side,
                ) + partial_realized
            )
            trade.pnl_pct = side_mult * (
                (exit_price - float(trade.entry_price)) / float(trade.entry_price)
            )
            stop_distance = abs(float(trade.entry_price) - float(trade.stop_price)) if trade.stop_price else 1
            trade.r_multiple = (
                side_mult * (exit_price - float(trade.entry_price)) / stop_distance
                if stop_distance > 0
                else 0
            )
        if exit_fee > 0:
            trade.commission = float(trade.commission or 0.0) + exit_fee

        try:
            session.commit()
        except Exception:
            session.rollback()

        log_trade(
            {
                "event": "exit",
                "trade_id": trade_id,
                "pair": pair,
                "exit_price": exit_price,
                "exit_reason": reason,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "r_multiple": trade.r_multiple,
                "commission": trade.commission,
            }
        )

        logger.info(
            f"Trade closed: {trade_id} | {pair} | "
            f"Exit: {exit_price:.6f} | Reason: {reason} | "
            f"PnL: {trade.pnl:.4f} ({trade.pnl_pct:.4%}) | R: {trade.r_multiple:.2f}"
        )

        self._trailing_extreme.pop(trade_id, None)
        return trade_id

    def _get_current_atr(self, pair: str) -> float:
        """Get current ATR for position management."""
        try:
            df = self.fetcher.fetch_klines(pair, "15min")
            if df.empty or len(df) < 15:
                return 0.0
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift(1)).abs()
            low_close = (df["low"] - df["close"].shift(1)).abs()
            tr = high_low.combine(high_close, max).combine(low_close, max)
            return float(tr.tail(14).mean())
        except Exception:
            return 0.0

    def _update_stop_order(self, trade: Trade, new_stop: float) -> None:
        """Update stop-loss order on exchange."""
        try:
            self._cancel_all_orders(trade.pair)
            side = str(trade.side or "BUY").upper()
            stop_side = "sell" if side == "BUY" else "buy"
            target_side = "sell" if side == "BUY" else "buy"
            self.executor.place_stop_order(trade.pair, stop_side, float(trade.quantity), new_stop)
            self.executor.place_limit_order(
                trade.pair,
                target_side,
                float(trade.quantity),
                float(trade.target_price),
            )
        except Exception as e:
            logger.error(f"Error updating stop order: {e}")

    def _cancel_all_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol."""
        self.executor.cancel_all_orders(symbol)

    def force_close_all(self) -> None:
        """Force close all open positions (dashboard kill switch)."""
        session = get_session()
        try:
            open_trades = session.query(Trade).filter(Trade.status == "open").all()
            for trade in open_trades:
                try:
                    ticker = self.fetcher.get_ticker(trade.pair)
                    self._close_trade(trade, float(ticker["price"]), "manual_force_close", session)
                except Exception as e:
                    logger.error(f"Error force-closing {trade.trade_id}: {e}")
        finally:
            session.close()

    def force_close_trade(self, trade_identifier: str | int) -> Optional[str]:
        """Force-close a specific open trade by trade_id or numeric id."""
        session = get_session()
        try:
            query = session.query(Trade).filter(Trade.status == "open")
            if isinstance(trade_identifier, int) or str(trade_identifier).isdigit():
                trade = query.filter(Trade.id == int(trade_identifier)).first()
            else:
                trade = query.filter(Trade.trade_id == str(trade_identifier)).first()

            if not trade:
                return None

            ticker = self.fetcher.get_ticker(trade.pair)
            return self._close_trade(trade, float(ticker["price"]), "manual_force_close", session)
        finally:
            session.close()
