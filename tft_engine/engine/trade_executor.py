"""
Trade persistence and execution state transitions.

This module intentionally focuses on persistence and orchestration points. Actual
exchange execution adapters can be injected later without changing API contracts.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from tft_engine.database.models import Trade, TradeFeature

logger = logging.getLogger(__name__)


class TradeExecutor:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def open_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        fees: float,
        ai_confidence: float,
        ai_score: float,
        model_version: str,
        feature_payload: dict,
    ) -> Trade:
        session = self._session_factory()
        try:
            trade = Trade(
                symbol=symbol,
                side=side.upper(),
                entry_price=float(entry_price),
                quantity=float(quantity),
                fees=float(fees),
                open_timestamp=datetime.utcnow(),
                ai_confidence=float(ai_confidence),
                ai_score=float(ai_score),
                model_version=model_version,
                win=None,
            )
            session.add(trade)
            session.flush()

            feature = TradeFeature(
                trade_id=trade.id,
                rsi=feature_payload.get("rsi"),
                ema_20=feature_payload.get("ema_20"),
                ema_50=feature_payload.get("ema_50"),
                atr=feature_payload.get("atr"),
                volatility=feature_payload.get("volatility"),
                volume=feature_payload.get("volume"),
                macd=feature_payload.get("macd"),
                market_regime=feature_payload.get("market_regime"),
            )
            session.add(feature)
            session.commit()
            session.refresh(trade)
            logger.info(f"Opened trade id={trade.id} symbol={symbol} qty={quantity:.6f}")
            return trade
        except Exception:
            session.rollback()
            logger.exception(f"Failed to open trade for {symbol}")
            raise
        finally:
            session.close()

    def close_trade(self, trade_id: int, exit_price: float, fees: float = 0.0) -> Trade:
        session = self._session_factory()
        try:
            trade = session.get(Trade, trade_id)
            if trade is None:
                raise ValueError(f"Trade {trade_id} not found")
            if trade.close_timestamp is not None:
                return trade

            trade.exit_price = float(exit_price)
            trade.fees = float(trade.fees or 0.0) + float(fees)
            trade.close_timestamp = datetime.utcnow()
            gross_pnl = (trade.exit_price - trade.entry_price) * trade.quantity
            if trade.side.upper() == "SELL":
                gross_pnl *= -1
            trade.pnl = gross_pnl - trade.fees
            trade.holding_time = (trade.close_timestamp - trade.open_timestamp).total_seconds()
            trade.win = 1 if (trade.pnl or 0.0) > 0 else 0
            session.commit()
            session.refresh(trade)
            logger.info(f"Closed trade id={trade.id} pnl={(trade.pnl or 0.0):.4f}")
            return trade
        except Exception:
            session.rollback()
            logger.exception(f"Failed to close trade id={trade_id}")
            raise
        finally:
            session.close()

    def get_open_trades(self) -> list[Trade]:
        session = self._session_factory()
        try:
            return (
                session.execute(
                    select(Trade).where(Trade.close_timestamp.is_(None)).order_by(Trade.open_timestamp.asc())
                )
                .scalars()
                .all()
            )
        finally:
            session.close()

    def latest_closed_trade(self) -> Optional[Trade]:
        session = self._session_factory()
        try:
            return (
                session.execute(
                    select(Trade)
                    .where(Trade.close_timestamp.is_not(None))
                    .order_by(Trade.close_timestamp.desc())
                )
                .scalars()
                .first()
            )
        finally:
            session.close()
