"""
Daily retraining pipeline.

Cron example (02:15 UTC daily):
15 2 * * * /usr/bin/python -m tft_engine.scripts.retrain_daily >> /var/log/tft_retrain.log 2>&1
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import select

from tft_engine.ai.metrics import max_drawdown, sharpe_ratio, win_rate
from tft_engine.ai.model_registry import ModelRegistryService
from tft_engine.ai.supervised.trainer import train_supervised_model
from tft_engine.ai.rl.train_rl import train_rl_from_database
from tft_engine.database.connection import get_session
from tft_engine.database.migrations import initialize_database
from tft_engine.database.models import AIMetric, Trade, TradeFeature

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrain_daily")


def _load_closed_trades(session) -> pd.DataFrame:
    rows = (
        session.execute(
            select(
                Trade.id,
                Trade.pnl,
                Trade.win,
                Trade.ai_confidence,
                Trade.open_timestamp,
                Trade.close_timestamp,
                TradeFeature.rsi,
                TradeFeature.ema_20,
                TradeFeature.ema_50,
                TradeFeature.atr,
                TradeFeature.volatility,
                TradeFeature.volume,
                TradeFeature.macd,
                TradeFeature.market_regime,
            )
            .join(TradeFeature, TradeFeature.trade_id == Trade.id, isouter=True)
            .where(Trade.close_timestamp.is_not(None))
            .order_by(Trade.close_timestamp.asc())
        )
        .mappings()
        .all()
    )
    if not rows:
        raise ValueError("No closed trades available for retraining.")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["open_timestamp"], utc=True)
    df["volume_change"] = df["volume"].pct_change().fillna(0.0)
    df["market_regime_encoded"] = df["market_regime"].fillna("sideways").map({"bear": 0, "sideways": 1, "bull": 2})
    df["hour"] = df["timestamp"].dt.hour.astype(float)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(float)
    df["win"] = df["win"].fillna((df["pnl"] > 0).astype(int)).astype(int)
    return df.fillna(0.0)


def _store_daily_ai_metrics(session, df: pd.DataFrame) -> AIMetric:
    pnls = df["pnl"].astype(float).to_list()
    confidence = df["ai_confidence"].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_list()
    cumulative_curve = np.cumsum(pnls).tolist() if pnls else [0.0]

    row = session.query(AIMetric).filter(AIMetric.date == date.today()).first()
    if row is None:
        row = AIMetric(date=date.today())
        session.add(row)

    row.win_rate = float(win_rate(pnls))
    row.sharpe_ratio = float(sharpe_ratio(pnls))
    row.max_drawdown = float(max_drawdown(cumulative_curve))
    row.avg_confidence = float(np.mean(confidence)) if confidence else 0.0
    row.cumulative_return = float(cumulative_curve[-1]) if cumulative_curve else 0.0
    session.commit()
    session.refresh(row)
    return row


def run_pipeline(force_rl: bool = False, rl_weekday: int = 0) -> dict:
    initialize_database()
    registry = ModelRegistryService(get_session)
    session = get_session()
    try:
        trades_df = _load_closed_trades(session)
        metrics = train_supervised_model(trades_df)
        registry.register_model(
            model_type="xgboost",
            version=metrics["version"],
            path="models/supervised/latest_xgb.pkl",
            metrics=metrics,
        )
        ai_metrics = _store_daily_ai_metrics(session, trades_df)

        rl_trained = False
        if force_rl or datetime.utcnow().weekday() == rl_weekday:
            rl_metrics = train_rl_from_database(get_session)
            registry.register_model(
                model_type="rl",
                version=rl_metrics["version"],
                path="models/rl/latest_rl.zip",
                metrics=rl_metrics,
            )
            rl_trained = True

        return {
            "date": str(ai_metrics.date),
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "xgb_version": metrics["version"],
            "rl_trained": rl_trained,
            "auto_reload": "enabled via model_registry polling every 5 minutes",
        }
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily retraining pipeline for TFT v2.")
    parser.add_argument("--force-rl", action="store_true", help="Train RL regardless of weekday schedule.")
    parser.add_argument(
        "--rl-weekday",
        type=int,
        default=0,
        help="Weekday for RL retraining (0=Monday .. 6=Sunday).",
    )
    args = parser.parse_args()
    try:
        result = run_pipeline(force_rl=args.force_rl, rl_weekday=args.rl_weekday)
        logger.info(f"Retraining completed: {result}")
        return 0
    except Exception:
        logger.exception("Retraining pipeline failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
