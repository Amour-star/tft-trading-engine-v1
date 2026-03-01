"""
Inference service for supervised trade scoring.
"""
from __future__ import annotations

from tft_engine.ai.supervised.features import features_from_snapshot
from tft_engine.ai.supervised.model import SupervisedPrediction, XGBoostTradeModel


class SupervisedInferenceService:
    def __init__(self, model_path: str | None = None) -> None:
        self.model = XGBoostTradeModel(model_path=model_path)
        self._loaded = False

    def load_model(self) -> None:
        self.model.load_model()
        self._loaded = True

    def predict(self, market_snapshot: dict) -> SupervisedPrediction:
        if not self._loaded:
            self.load_model()
        features = features_from_snapshot(market_snapshot)
        return self.model.predict(features)

    @property
    def model_version(self) -> str:
        return self.model.model_version

