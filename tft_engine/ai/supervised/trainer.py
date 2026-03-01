"""
Training orchestration for the XGBoost supervised model.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tft_engine.ai.supervised.features import FEATURE_COLUMNS, build_feature_matrix
from tft_engine.ai.supervised.model import XGBoostTradeModel

logger = logging.getLogger(__name__)


def train_supervised_model(
    ohlcv_or_feature_df: pd.DataFrame,
    model_path: str | None = None,
) -> dict[str, Any]:
    if "win" in ohlcv_or_feature_df.columns:
        dataset = ohlcv_or_feature_df.copy()
    else:
        dataset = build_feature_matrix(ohlcv_or_feature_df)
        if "pnl" in dataset.columns and "win" not in dataset.columns:
            dataset["win"] = (dataset["pnl"] > 0).astype(int)

    if "win" not in dataset.columns:
        raise ValueError("Training dataset must include `win` or `pnl` columns.")

    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["win"])
    if len(dataset) < 100:
        raise ValueError("Insufficient training rows. Minimum recommended: 100.")

    x = dataset[FEATURE_COLUMNS].astype(float).values
    y = dataset["win"].astype(int).values
    returns = dataset["pnl"].astype(float).values if "pnl" in dataset.columns else None

    if returns is not None:
        x_train, x_test, y_train, y_test, returns_train, _ = train_test_split(
            x,
            y,
            returns,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        returns_train = None

    model = XGBoostTradeModel(model_path=model_path)
    version = model.train_model(x_train, y_train, FEATURE_COLUMNS, returns_train=returns_train)
    model.save_model()

    probas = model.classifier.predict_proba(x_test)[:, 1]
    preds = (probas >= 0.5).astype(int)

    metrics = {
        "version": version,
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, probas)) if len(np.unique(y_test)) > 1 else 0.0,
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "samples": int(len(dataset)),
    }
    logger.info(f"Supervised model trained: {metrics}")
    return metrics
