"""
Train XGBoost meta-model from closed trade outcomes.
"""
import sys

sys.path.insert(0, ".")

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from loguru import logger

from data.database import Trade, get_session, init_db, register_model_version
from models.meta_model import XGBoostMetaModel
from utils.logging import setup_logging


def _import_xgboost():
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ImportError("xgboost is required. Install with `pip install xgboost`.") from exc
    return XGBClassifier


def build_training_set(limit: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    session = get_session()
    try:
        rows = (
            session.query(Trade)
            .filter(Trade.status == "closed", Trade.pnl.isnot(None))
            .order_by(Trade.exit_time.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()

    builder = XGBoostMetaModel()
    samples = []
    labels = []

    for trade in rows:
        prediction = trade.prediction or {}
        features = trade.features_at_entry or {}
        ts = trade.entry_time or datetime.utcnow()

        feature_row = builder.build_features(prediction, features, ts)
        samples.append([float(feature_row.get(c, 0.0)) for c in builder.default_feature_columns()])
        labels.append(1 if float(trade.pnl or 0.0) > 0 else 0)

    if not samples:
        return np.empty((0, 0)), np.empty((0,)), []

    return np.asarray(samples, dtype=float), np.asarray(labels, dtype=int), builder.default_feature_columns()


def main():
    setup_logging()
    init_db()

    parser = argparse.ArgumentParser(description="Train XGBoost meta-model")
    parser.add_argument("--limit", type=int, default=5000, help="Max closed trades to use")
    parser.add_argument("--path", type=str, default="models/meta/latest_xgb.pkl", help="Model output path")
    args = parser.parse_args()

    x_train, y_train, columns = build_training_set(args.limit)
    if x_train.size == 0 or y_train.size == 0:
        logger.error("No eligible closed trades found for meta-model training.")
        return

    if len(np.unique(y_train)) < 2:
        logger.error("Need both winning and losing samples to train meta-model.")
        return

    XGBClassifier = _import_xgboost()
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=42,
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_train)
    accuracy = float((preds == y_train).mean())
    version = f"xgb_meta_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": columns,
        "model_version": version,
        "trained_at": datetime.utcnow().isoformat(),
        "accuracy": accuracy,
        "samples": int(len(y_train)),
    }
    joblib.dump(artifact, path)

    register_model_version(
        model_type="xgb_meta",
        version=version,
        path=str(path),
        model_metadata={"accuracy": accuracy, "samples": int(len(y_train))},
        activate=True,
    )

    logger.info(f"Meta-model trained: version={version} samples={len(y_train)} accuracy={accuracy:.4f}")
    logger.info(f"Saved artifact: {path}")


if __name__ == "__main__":
    main()
