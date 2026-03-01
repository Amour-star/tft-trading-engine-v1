"""
Stable-Baselines3 PPO agent wrapper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tft_engine.config import config


def _import_ppo():
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise ImportError(
            "stable-baselines3 is required. Install with `pip install stable-baselines3`."
        ) from exc
    return PPO


class RLTradeAgent:
    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path or config.rl_model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None

    def load_model(self) -> None:
        PPO = _import_ppo()
        if not self.model_path.exists():
            raise FileNotFoundError(f"RL model not found: {self.model_path}")
        self.model = PPO.load(str(self.model_path))

    def train(self, env, total_timesteps: int = 100_000) -> None:
        PPO = _import_ppo()
        self.model = PPO("MlpPolicy", env, verbose=0, n_steps=2048, batch_size=128, learning_rate=3e-4)
        self.model.learn(total_timesteps=total_timesteps)

    def save_model(self) -> None:
        if self.model is None:
            raise RuntimeError("RL model is not trained; cannot save.")
        self.model.save(str(self.model_path))

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        if self.model is None:
            self.load_model()
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

