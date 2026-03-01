"""
Train or retrain the TFT model.
Loads historical data, trains, validates, and saves the model.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import XRP_ONLY_SYMBOL, settings
from data.database import ModelMetric, get_session, init_db, register_model_version
from models.tft_model import train_tft
from utils.logging import setup_logging

DATA_DIR = ROOT / "data" / "historical"


def main() -> None:
    setup_logging()
    init_db()

    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument("--timeframe", default="15min", help="Primary timeframe")
    parser.add_argument("--name", type=str, help="Model version name")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--quick", action="store_true", help="Quick smoke training mode")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing model")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Optional symbol whitelist for training universe (example: XRP-USDT)",
    )
    parser.add_argument(
        "--required-symbols",
        nargs="+",
        default=[XRP_ONLY_SYMBOL],
        help="Symbols that must be present in training data and model vocabulary",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TFT MODEL TRAINING")
    logger.info("=" * 60)

    data_files = sorted(DATA_DIR.glob(f"*_{args.timeframe}.parquet"))
    if args.symbols:
        allow = {str(symbol).strip() for symbol in args.symbols if str(symbol).strip()}
        data_files = [
            file_path
            for file_path in data_files
            if file_path.stem.replace(f"_{args.timeframe}", "") in allow
        ]
        logger.info(f"Training symbol filter enabled: {sorted(allow)}")
    if not data_files:
        raise RuntimeError(
            f"No data files found in {DATA_DIR} for timeframe {args.timeframe}. "
            "Run scripts/fetch_history.py first."
        )

    logger.info(f"Found {len(data_files)} data files")

    all_dfs: list[pd.DataFrame] = []
    for file_path in data_files:
        try:
            df = pd.read_parquet(file_path)
        except Exception as exc:
            logger.warning(f"Skipping unreadable file {file_path}: {exc}")
            continue
        if df.empty:
            continue
        if "pair" not in df.columns:
            df["pair"] = file_path.stem.replace(f"_{args.timeframe}", "")
        all_dfs.append(df)
        logger.info(f"Loaded {file_path.stem}: {len(df)} rows")

    if not all_dfs:
        raise RuntimeError("No usable training files found.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values(["pair", "timestamp"], inplace=True, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} rows, {combined['pair'].nunique()} pairs")
    required_symbols = sorted({str(symbol).strip() for symbol in args.required_symbols if str(symbol).strip()})
    combined_pairs = {str(pair).strip() for pair in combined["pair"].dropna().unique()}
    missing_required = sorted(set(required_symbols).difference(combined_pairs))
    if missing_required:
        raise RuntimeError(
            "Training corpus is missing required symbol(s): "
            f"{missing_required}. Available pairs: {sorted(combined_pairs)}"
        )
    logger.info(f"Required symbols: {required_symbols}")

    if args.quick:
        args.epochs = 1 if args.epochs is None else min(args.epochs, 1)
        args.batch_size = args.batch_size or min(settings.model.batch_size, 32)
        logger.info("Quick mode enabled: forcing 1 epoch and reduced batch size.")

    version_name = args.name or f"tft_v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    version, metrics = train_tft(
        combined,
        model_name=version_name,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        required_pairs=required_symbols,
    )

    session = get_session()
    try:
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
                "batch_size": args.batch_size or settings.model.batch_size,
                "max_epochs": args.epochs or settings.model.max_epochs,
            },
        )
        session.add(model_metric)
        session.commit()
        register_model_version(
            model_type="tft",
            version=version,
            path=str(Path("saved_models") / version),
            model_metadata=metrics,
            activate=True,
        )
        logger.info(f"Model {version} saved and activated")
    except Exception as exc:
        session.rollback()
        logger.error(f"Error saving model metrics: {exc}")
        raise
    finally:
        session.close()

    logger.info("Training complete!")
    logger.info(f"Model version: {version}")
    logger.info(f"Validation loss: {metrics.get('validation_loss', 'N/A')}")


if __name__ == "__main__":
    main()
