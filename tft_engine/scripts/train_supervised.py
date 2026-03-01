"""
CLI script to train the supervised XGBoost model from SQLite trade data.
"""
from __future__ import annotations

import logging
import sys

import pandas as pd
from sqlalchemy import select

from tft_engine.ai.model_registry import ModelRegistryService
from tft_engine.ai.supervised.trainer import train_supervised_model
from tft_engine.database.connection import get_session
from tft_engine.database.migrations import initialize_database
from tft_engine.database.models import Trade, TradeFeature

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_supervised")


def load_training_frame() -> pd.DataFrame:
    session = get_session()
    try:
        rows = (
            session.execute(
                select(
                    Trade.id,
                    Trade.pnl,
                    Trade.win,
                    Trade.open_timestamp,
                    TradeFeature.rsi,
                    TradeFeature.ema_20,
                    TradeFeature.ema_50,
                    TradeFeature.atr,
                    TradeFeature.volatility,
                    TradeFeature.macd,
                    TradeFeature.market_regime,
                    TradeFeature.volume,
                )
                .join(TradeFeature, TradeFeature.trade_id == Trade.id, isouter=True)
                .where(Trade.close_timestamp.is_not(None))
            )
            .mappings()
            .all()
        )
    finally:
        session.close()

    if not rows:
        raise ValueError("No closed trades found for supervised training.")

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["open_timestamp"], utc=True)
    frame["volume_change"] = frame["volume"].pct_change().fillna(0.0)
    frame["market_regime_encoded"] = frame["market_regime"].fillna("sideways").map(
        {"bear": 0, "sideways": 1, "bull": 2}
    )
    frame["hour"] = frame["timestamp"].dt.hour.astype(float)
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek.astype(float)
    frame["win"] = frame["win"].fillna((frame["pnl"] > 0).astype(int)).astype(int)
    frame = frame.fillna(0.0)
    return frame


def main() -> int:
    try:
        initialize_database()
        df = load_training_frame()
        metrics = train_supervised_model(df)
        registry = ModelRegistryService(get_session)
        registry.register_model(
            model_type="xgboost",
            version=metrics["version"],
            path="models/supervised/latest_xgb.pkl",
            metrics=metrics,
        )
        logger.info(f"Supervised training complete: {metrics}")
        return 0
    except Exception:
        logger.exception("Supervised training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
