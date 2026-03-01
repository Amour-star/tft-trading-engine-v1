"""
RL training entrypoints.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import select

from tft_engine.ai.rl.agent import RLTradeAgent
from tft_engine.ai.rl.environment import TradeManagementEnv
from tft_engine.database.models import Trade, TradeFeature

logger = logging.getLogger(__name__)


def _build_market_frame(session) -> pd.DataFrame:
    rows = (
        session.execute(
            select(
                Trade.open_timestamp,
                Trade.entry_price,
                Trade.exit_price,
                Trade.pnl,
                TradeFeature.rsi,
                TradeFeature.ema_20,
                TradeFeature.volatility,
            )
            .join(TradeFeature, TradeFeature.trade_id == Trade.id, isouter=True)
            .where(Trade.close_timestamp.is_not(None))
            .order_by(Trade.open_timestamp.asc())
        )
        .mappings()
        .all()
    )
    if not rows:
        raise ValueError("No closed trades available for RL training.")

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["open_timestamp"], utc=True)
    frame["close"] = frame["entry_price"].astype(float)
    frame["rsi"] = frame["rsi"].fillna(50.0).astype(float)
    frame["ema_20"] = frame["ema_20"].fillna(frame["close"]).astype(float)
    frame["volatility"] = frame["volatility"].fillna(0.0).astype(float)
    return frame[["timestamp", "close", "rsi", "ema_20", "volatility"]]


def train_rl_from_database(session_factory, model_path: str | None = None, timesteps: int = 75_000) -> dict:
    session = session_factory()
    try:
        market_df = _build_market_frame(session)
    finally:
        session.close()

    env = TradeManagementEnv(market_df)
    agent = RLTradeAgent(model_path=model_path)
    agent.train(env, total_timesteps=timesteps)
    agent.save_model()

    return {
        "version": f"rl_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "timesteps": timesteps,
        "samples": int(len(market_df)),
    }

