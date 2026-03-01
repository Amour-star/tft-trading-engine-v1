"""
XGBoost supervised model wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from tft_engine.config import config


def _import_xgboost():
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception as exc:
        raise ImportError(
            "xgboost is required. Install with `pip install xgboost`."
        ) from exc
    return XGBClassifier, XGBRegressor


@dataclass
class SupervisedPrediction:
    win_probability: float
    expected_return: float
    confidence_score: float
    model_version: str


class XGBoostTradeModel:
    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path or config.supervised_model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.classifier = None
        self.return_regressor = None
        self.feature_columns: list[str] = []
        self.model_version = "unloaded"
        self.avg_win_return = 0.01
        self.avg_loss_return = -0.01

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Supervised model not found: {self.model_path}")
        artifact = joblib.load(self.model_path)
        self.classifier = artifact["classifier"]
        self.return_regressor = artifact.get("return_regressor")
        self.feature_columns = artifact["feature_columns"]
        self.model_version = artifact.get("model_version", "unknown")
        self.avg_win_return = float(artifact.get("avg_win_return", 0.01))
        self.avg_loss_return = float(artifact.get("avg_loss_return", -0.01))

    def save_model(self) -> str:
        if self.classifier is None:
            raise RuntimeError("Model is not trained; cannot save.")
        if not self.model_version or self.model_version == "unloaded":
            self.model_version = f"xgb_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        artifact = {
            "classifier": self.classifier,
            "return_regressor": self.return_regressor,
            "feature_columns": self.feature_columns,
            "model_version": self.model_version,
            "avg_win_return": self.avg_win_return,
            "avg_loss_return": self.avg_loss_return,
        }
        joblib.dump(artifact, self.model_path)
        return self.model_version

    def train_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        feature_columns: list[str],
        returns_train: Optional[np.ndarray] = None,
    ) -> str:
        XGBClassifier, XGBRegressor = _import_xgboost()
        clf = XGBClassifier(
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
        clf.fit(x_train, y_train)
        self.classifier = clf
        self.feature_columns = feature_columns

        if returns_train is not None and returns_train.size > 10:
            reg = XGBRegressor(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=4,
                random_state=42,
            )
            reg.fit(x_train, returns_train)
            self.return_regressor = reg

        wins = returns_train[returns_train > 0] if returns_train is not None else np.array([])
        losses = returns_train[returns_train <= 0] if returns_train is not None else np.array([])
        if wins.size > 0:
            self.avg_win_return = float(np.mean(wins))
        if losses.size > 0:
            self.avg_loss_return = float(np.mean(losses))

        self.model_version = f"xgb_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        return self.model_version

    def predict(self, features: dict) -> SupervisedPrediction:
        if self.classifier is None:
            self.load_model()
        x = np.asarray([float(features.get(col, 0.0)) for col in self.feature_columns], dtype=float).reshape(1, -1)
        win_probability = float(self.classifier.predict_proba(x)[0, 1])

        if self.return_regressor is not None:
            expected_return = float(self.return_regressor.predict(x)[0])
        else:
            expected_return = win_probability * self.avg_win_return + (1.0 - win_probability) * self.avg_loss_return

        # Confidence score is calibrated to [0,1] where extreme probabilities are more confident.
        confidence_score = float(abs(win_probability - 0.5) * 2.0)
        return SupervisedPrediction(
            win_probability=win_probability,
            expected_return=expected_return,
            confidence_score=confidence_score,
            model_version=self.model_version,
        )

