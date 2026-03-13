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


def walk_forward_score(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    folds: int,
    classifier_factory,
) -> dict:
    sample_count = len(y_train)
    if sample_count < 40 or folds < 2:
        return {"folds": 0, "accuracy": 0.0, "evaluated_samples": 0}

    fold_boundaries = np.linspace(0, sample_count, folds + 1, dtype=int)
    accuracies: list[float] = []
    evaluated_samples = 0
    for fold_idx in range(1, len(fold_boundaries) - 1):
        train_end = fold_boundaries[fold_idx]
        test_end = fold_boundaries[fold_idx + 1]
        x_fit = x_train[:train_end]
        y_fit = y_train[:train_end]
        x_test = x_train[train_end:test_end]
        y_test = y_train[train_end:test_end]
        if len(x_fit) < 20 or len(x_test) == 0 or len(np.unique(y_fit)) < 2 or len(np.unique(y_test)) < 2:
            continue
        model = classifier_factory()
        model.fit(x_fit, y_fit)
        preds = model.predict(x_test)
        accuracies.append(float((preds == y_test).mean()))
        evaluated_samples += int(len(y_test))

    if not accuracies:
        return {"folds": 0, "accuracy": 0.0, "evaluated_samples": 0}
    return {
        "folds": int(len(accuracies)),
        "accuracy": float(np.mean(accuracies)),
        "evaluated_samples": int(evaluated_samples),
    }


def main():
    setup_logging()
    init_db()

    parser = argparse.ArgumentParser(description="Train XGBoost meta-model")
    parser.add_argument("--limit", type=int, default=5000, help="Max closed trades to use")
    parser.add_argument(
        "--path",
        type=str,
        default="saved_models/meta/latest_xgb.pkl",
        help="Model output path",
    )
    parser.add_argument("--walk-forward-folds", type=int, default=4, help="Sequential validation folds")
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

    def _factory():
        return XGBClassifier(
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

    preds = model.predict(x_train)
    accuracy = float((preds == y_train).mean())
    walk_forward = walk_forward_score(
        x_train,
        y_train,
        folds=max(2, int(args.walk_forward_folds)),
        classifier_factory=_factory,
    )
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
        "walk_forward": walk_forward,
    }
    joblib.dump(artifact, path)

    register_model_version(
        model_type="xgb_meta",
        version=version,
        path=str(path),
        model_metadata={"accuracy": accuracy, "samples": int(len(y_train)), "walk_forward": walk_forward},
        activate=True,
    )

    logger.info(
        "Meta-model trained: version={} samples={} accuracy={:.4f} walk_forward_accuracy={:.4f} folds={}",
        version,
        len(y_train),
        accuracy,
        float(walk_forward.get("accuracy", 0.0)),
        int(walk_forward.get("folds", 0)),
    )
    logger.info(f"Saved artifact: {path}")


if __name__ == "__main__":
    main()
