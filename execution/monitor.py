"""
Position monitor: real-time trade management.
Monitors open positions and adjusts stops/exits based on market conditions.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger

from data.database import get_session, Trade
from data.fetcher import KuCoinDataFetcher
from data.features import compute_features
from execution.base_executor import BaseExecutor
from models.tft_model import TFTPredictor
from utils.logging import log_trade


class PositionMonitor:
    """
    Monitors open positions in real time.
    Implements trailing stops, confidence-based exits, and momentum-based management.
    """

    def __init__(
        self,
        fetcher: KuCoinDataFetcher,
        executor: BaseExecutor,
        predictor: TFTPredictor,
    ) -> None:
        self.fetcher = fetcher
        self.executor = executor
        self.predictor = predictor
        self._trailing_high: float = 0.0

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
            entry_price = trade.entry_price
            stop_price = trade.stop_price
            target_price = trade.target_price
            quantity = trade.quantity

            # Get current price
            try:
                ticker = self.fetcher.get_ticker(pair)
                current_price = ticker["price"]
            except Exception as e:
                logger.error(f"Cannot get price for {pair}: {e}")
                return None

            # Update trailing high
            if current_price > self._trailing_high:
                self._trailing_high = current_price

            # ---- Check 1: Has price hit stop or target? ----
            if current_price <= stop_price:
                return self._close_trade(trade, current_price, "stop", session)

            if current_price >= target_price:
                return self._close_trade(trade, current_price, "target", session)

            # ---- Check 2: Trailing stop (if in profit) ----
            if current_price > entry_price:
                pnl_pct = (current_price - entry_price) / entry_price
                if pnl_pct > 0.02:  # Trail after 2% profit
                    atr = self._get_current_atr(pair)
                    trail_stop = self._trailing_high - (atr * 1.5)
                    if trail_stop > stop_price:
                        logger.info(f"Trailing stop: {stop_price:.6f} → {trail_stop:.6f}")
                        trade.stop_price = trail_stop
                        stop_price = trail_stop
                        # Update stop order on exchange
                        self._update_stop_order(trade, trail_stop)

            # ---- Check 3: Model confidence check ----
            try:
                df = self.fetcher.fetch_klines(pair, "15min")
                if not df.empty and len(df) > 100:
                    df["pair"] = pair
                    btc_df = self.fetcher.fetch_klines("BTC-USDT", "15min")
                    df = compute_features(df, btc_df)
                    prediction = self.predictor.predict(df, pair)

                    new_confidence = prediction.get("confidence", 0)
                    prob_down = prediction.get("prob_down", 0)

                    # If confidence drops significantly, tighten stop
                    if new_confidence < 0.4:
                        tighter_stop = current_price - (current_price - stop_price) * 0.5
                        if tighter_stop > stop_price:
                            logger.info(f"Tightening stop on low confidence: {new_confidence:.3f}")
                            trade.stop_price = tighter_stop
                            self._update_stop_order(trade, tighter_stop)

                    # If strong opposite signal, close
                    if prob_down > 0.75 and new_confidence > 0.6:
                        logger.info(f"Strong bearish signal: prob_down={prob_down:.3f}")
                        return self._close_trade(trade, current_price, "signal_reversal", session)

            except Exception as e:
                logger.warning(f"Model check failed: {e}")

            # ---- Check 4: Momentum weakening ----
            try:
                df = self.fetcher.fetch_klines(pair, "5min")
                if not df.empty and len(df) > 20:
                    recent_returns = df["close"].pct_change().tail(5)
                    if all(r < 0 for r in recent_returns.dropna()):
                        # 5 consecutive negative 5min candles
                        logger.info("Momentum weakening: 5 consecutive red candles")
                        pnl_pct = (current_price - entry_price) / entry_price
                        if pnl_pct > 0.005:  # Only exit early if slightly in profit
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
        quantity = trade.quantity

        # Market close on exchange
        try:
            # Cancel existing orders first
            self._cancel_all_orders(pair)
            # Close position through executor (live order / paper simulation)
            close_result = self.executor.sell(pair, quantity, price=exit_price)
            exit_price = float(close_result.get("fill_price", exit_price))
            exit_fee = float(close_result.get("fee", 0.0))
            realized_pnl = close_result.get("realized_pnl")
        except Exception as e:
            logger.error(f"Error closing position on exchange: {e}")
            exit_fee = 0.0
            realized_pnl = None

        # Update database
        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow()
        trade.exit_reason = reason
        trade.status = "closed"

        if trade.entry_price:
            trade.pnl = (
                float(realized_pnl)
                if realized_pnl is not None
                else (exit_price - trade.entry_price) * quantity
            )
            trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
            stop_distance = abs(trade.entry_price - trade.stop_price) if trade.stop_price else 1
            trade.r_multiple = (exit_price - trade.entry_price) / stop_distance if stop_distance > 0 else 0
        if exit_fee > 0:
            trade.commission = (trade.commission or 0.0) + exit_fee

        try:
            session.commit()
        except Exception:
            session.rollback()

        log_trade({
            "event": "exit",
            "trade_id": trade_id,
            "pair": pair,
            "exit_price": exit_price,
            "exit_reason": reason,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "r_multiple": trade.r_multiple,
            "commission": trade.commission,
        })

        logger.info(
            f"Trade closed: {trade_id} | {pair} | "
            f"Exit: {exit_price:.6f} | Reason: {reason} | "
            f"PnL: {trade.pnl:.4f} ({trade.pnl_pct:.4%}) | R: {trade.r_multiple:.2f}"
        )

        # Reset trailing
        self._trailing_high = 0.0

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
            # Re-place stop
            self.executor.place_stop_order(trade.pair, "sell", trade.quantity, new_stop)
            # Re-place target
            self.executor.place_limit_sell(trade.pair, trade.quantity, trade.target_price)
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
                    self._close_trade(trade, ticker["price"], "manual_force_close", session)
                except Exception as e:
                    logger.error(f"Error force-closing {trade.trade_id}: {e}")
        finally:
            session.close()
