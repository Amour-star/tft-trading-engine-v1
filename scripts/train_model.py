"""
Train or retrain the TFT model.
Loads historical data, trains, validates, and saves the model.
"""
import sys
sys.path.insert(0, ".")

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import settings
from data.database import init_db, get_session, ModelMetric
from models.tft_model import train_tft, TFTPredictor
from utils.logging import setup_logging

DATA_DIR = Path("data/historical")


def main():
    setup_logging()
    init_db()

    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument("--timeframe", default="15min", help="Primary timeframe")
    parser.add_argument("--name", type=str, help="Model version name")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing model")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TFT MODEL TRAINING")
    logger.info("=" * 60)

    # Load all data files for the specified timeframe
    data_files = list(DATA_DIR.glob(f"*_{args.timeframe}.parquet"))
    if not data_files:
        logger.error(f"No data files found in {DATA_DIR} for timeframe {args.timeframe}")
        logger.error("Run scripts/fetch_history.py first")
        return

    logger.info(f"Found {len(data_files)} data files")

    # Combine all pair data
    all_dfs = []
    for f in data_files:
        df = pd.read_parquet(f)
        if "pair" not in df.columns:
            df["pair"] = f.stem.replace(f"_{args.timeframe}", "")
        all_dfs.append(df)
        logger.info(f"Loaded {f.stem}: {len(df)} rows")

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} rows, {combined['pair'].nunique()} pairs")

    # Train
    version_name = args.name or f"tft_v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    version, metrics = train_tft(combined, model_name=version_name)

    # Save model metrics to database
    session = get_session()
    try:
        # Deactivate previous models
        session.query(ModelMetric).update({ModelMetric.is_active: False})

        model_metric = ModelMetric(
            model_version=version,
            trained_at=datetime.utcnow(),
            training_loss=metrics.get("training_loss"),
            validation_loss=metrics.get("validation_loss"),
            is_active=True,
            hyperparameters={
                "encoder_length": settings.model.encoder_length,
                "prediction_length": settings.model.prediction_length,
                "hidden_size": settings.model.hidden_size,
                "attention_head_size": settings.model.attention_head_size,
                "dropout": settings.model.dropout,
                "learning_rate": settings.model.learning_rate,
            },
        )
        session.add(model_metric)
        session.commit()
        logger.info(f"Model {version} saved and activated")
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving model metrics: {e}")
    finally:
        session.close()

    logger.info("Training complete!")
    logger.info(f"Model version: {version}")
    logger.info(f"Validation loss: {metrics.get('validation_loss', 'N/A')}")


if __name__ == "__main__":
    main()
