"""
Inference utility for RL trade management decisions.
"""
from __future__ import annotations

import numpy as np

from tft_engine.ai.rl.agent import RLTradeAgent


class RLInferenceService:
    def __init__(self, model_path: str | None = None):
        self.agent = RLTradeAgent(model_path=model_path)
        self._loaded = False

    def load_model(self) -> None:
        self.agent.load_model()
        self._loaded = True

    def choose_action(self, observation: np.ndarray) -> int:
        if not self._loaded:
            self.load_model()
        return self.agent.act(observation, deterministic=True)

