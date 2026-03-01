"""
PPO-based position manager for sizing, stop adjustments, and exit timing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from data.database import get_latest_model_version


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _import_ppo():
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise ImportError(
            "stable-baselines3 is required. Install with `pip install stable-baselines3`."
        ) from exc
    return PPO


@dataclass
class RLAction:
    action: str
    raw_action: int
    size_multiplier: float
    stop_atr_multiplier: float
    should_exit: bool
    model_version: str


class PPOPositionManager:
    """
    Action space:
    0: increase position
    1: decrease position
    2: hold
    3: close
    """

    ACTION_MAP = {
        0: "increase",
        1: "decrease",
        2: "hold",
        3: "close",
    }

    SIZE_MULTIPLIERS = {
        0: 1.25,
        1: 0.70,
        2: 1.00,
        3: 0.0,
    }

    STOP_MULTIPLIERS = {
        0: 2.2,
        1: 1.4,
        2: 2.0,
        3: 1.0,
    }

    def __init__(self, model_path: str | None = None) -> None:
        default_path = Path(os.getenv("RL_MODEL_PATH", "models/rl/latest_ppo.zip"))
        self.model_path = Path(model_path) if model_path else default_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.model_version = "ppo_fallback"

    def _resolve_model_path(self) -> Path:
        try:
            latest = get_latest_model_version("ppo")
            if latest and latest.path:
                return Path(latest.path)
        except Exception:
            pass
        return self.model_path

    def load(self) -> bool:
        path = self._resolve_model_path()
        if not path.exists():
            self.model = None
            self.model_version = "ppo_fallback"
            return False

        PPO = _import_ppo()
        self.model = PPO.load(str(path))
        self.model_version = path.stem
        self.model_path = path
        return True

    def _observation(self, state: Dict[str, Any]) -> np.ndarray:
        current_price = _safe_float(state.get("current_price", state.get("entry_price", 0.0)))
        entry_price = _safe_float(state.get("entry_price", current_price))
        unrealized = _safe_float(state.get("unrealized_pnl", (current_price - entry_price) * _safe_float(state.get("quantity", 0.0))))
        obs = np.asarray(
            [
                current_price,
                _safe_float(state.get("rsi", state.get("rsi_14", 50.0))),
                _safe_float(state.get("ema_20", current_price)),
                _safe_float(state.get("volatility", state.get("volatility_20", 0.0))),
                _safe_float(state.get("quantity", 0.0)),
                unrealized,
                _safe_float(state.get("time_in_trade", 0.0)),
            ],
            dtype=np.float32,
        )
        return obs

    def _heuristic_action(self, state: Dict[str, Any]) -> int:
        unrealized = _safe_float(state.get("unrealized_pnl", 0.0))
        time_in_trade = _safe_float(state.get("time_in_trade", 0.0))
        volatility = _safe_float(state.get("volatility", state.get("volatility_20", 0.0)))

        if unrealized < 0 and volatility > 0.03:
            return 1
        if unrealized > 0 and time_in_trade > 30:
            return 1
        if unrealized < 0 and time_in_trade > 80:
            return 3
        return 2

    def _act(self, state: Dict[str, Any], deterministic: bool = True) -> int:
        if self.model is None and not self.load():
            return self._heuristic_action(state)

        obs = self._observation(state)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def initial_size(self, state: Dict[str, Any]) -> float:
        action = self._act(state, deterministic=True)
        return self.SIZE_MULTIPLIERS.get(action, 1.0)

    def step(self, state: Dict[str, Any]) -> RLAction:
        action = self._act(state, deterministic=True)
        return RLAction(
            action=self.ACTION_MAP.get(action, "hold"),
            raw_action=action,
            size_multiplier=self.SIZE_MULTIPLIERS.get(action, 1.0),
            stop_atr_multiplier=self.STOP_MULTIPLIERS.get(action, 2.0),
            should_exit=action == 3,
            model_version=self.model_version,
        )
