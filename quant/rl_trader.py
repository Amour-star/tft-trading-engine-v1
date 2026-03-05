from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple

from quant.config import QuantEngineConfig
from quant.types import RegimeState, StrategySignal


@dataclass
class RLDecision:
    action: str
    size_multiplier: float
    confidence_boost: float


class ReinforcementLearningTrader:
    """
    Lightweight Q-learning trader for entry/exit timing and sizing.
    Action-space aligns with institutional execution flow:
    hold, enter, scale, reduce, exit.
    """

    def __init__(self, cfg: QuantEngineConfig) -> None:
        self.cfg = cfg
        self.actions: List[str] = ["hold", "enter", "scale", "reduce", "exit"]
        self.q: DefaultDict[Tuple[str, int, int], Dict[str, float]] = defaultdict(
            lambda: {action: 0.0 for action in self.actions}
        )
        self.alpha = cfg.rl_learning_rate
        self.gamma = cfg.rl_discount_factor
        self.epsilon = cfg.rl_exploration

    def decide(self, signal: StrategySignal, regime: RegimeState, has_position: bool) -> RLDecision:
        state = self._state(signal, regime, has_position)
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            values = self.q[state]
            action = max(values, key=values.get)

        if action == "enter":
            size_multiplier = 1.0 + max(0.0, signal.confidence - 0.5) * 0.8
        elif action == "scale":
            size_multiplier = 1.25
        elif action == "reduce":
            size_multiplier = 0.55
        elif action == "exit":
            size_multiplier = 0.0
        else:
            size_multiplier = 1.0

        confidence_boost = max(-0.15, min(0.15, (self.q[state][action]) * 0.1))
        return RLDecision(
            action=action,
            size_multiplier=size_multiplier,
            confidence_boost=confidence_boost,
        )

    def learn(
        self,
        previous_signal: StrategySignal,
        previous_regime: RegimeState,
        previous_has_position: bool,
        action: str,
        reward: float,
        next_signal: StrategySignal,
        next_regime: RegimeState,
        next_has_position: bool,
    ) -> None:
        prev_state = self._state(previous_signal, previous_regime, previous_has_position)
        next_state = self._state(next_signal, next_regime, next_has_position)
        q_prev = self.q[prev_state].get(action, 0.0)
        q_next_max = max(self.q[next_state].values())
        td_target = reward + self.gamma * q_next_max
        self.q[prev_state][action] = q_prev + self.alpha * (td_target - q_prev)

    def _state(self, signal: StrategySignal, regime: RegimeState, has_position: bool) -> Tuple[str, int, int]:
        confidence_bucket = int(min(9, max(0, round(signal.confidence * 10))))
        direction_bucket = int(signal.direction)
        position_flag = 1 if has_position else 0
        regime_key = regime.label[:16]
        return regime_key, direction_bucket * 10 + confidence_bucket, position_flag
