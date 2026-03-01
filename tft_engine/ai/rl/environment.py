"""
Custom Gymnasium environment for trade management actions.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class EnvConfig:
    initial_capital: float = 10_000.0
    max_steps: int = 512
    drawdown_penalty: float = 0.5
    overtrade_penalty: float = 0.0005


class TradeManagementEnv(gym.Env):
    """
    Action space:
    0 -> Increase position
    1 -> Decrease position
    2 -> Hold
    3 -> Close trade
    """

    metadata = {"render_modes": []}

    def __init__(self, market_data: pd.DataFrame, config: EnvConfig | None = None):
        super().__init__()
        if len(market_data) < 3:
            raise ValueError("market_data requires at least 3 rows.")
        self.df = market_data.reset_index(drop=True)
        self.cfg = config or EnvConfig()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self._reset_state()

    def _reset_state(self) -> None:
        self.step_idx = 1
        self.position_size = 0.0
        self.entry_price = float(self.df.loc[self.step_idx, "close"])
        self.cash = self.cfg.initial_capital
        self.portfolio_value = self.cfg.initial_capital
        self.peak_value = self.cfg.initial_capital
        self.time_in_trade = 0.0
        self.last_action = 2

    def _current_price(self) -> float:
        return float(self.df.loc[self.step_idx, "close"])

    def _unrealized_pnl(self, price: float) -> float:
        if self.position_size <= 0:
            return 0.0
        return (price - self.entry_price) * self.position_size

    def _obs(self) -> np.ndarray:
        row = self.df.loc[self.step_idx]
        price = float(row.get("close", 0.0))
        obs = np.asarray(
            [
                price,
                float(row.get("rsi", 50.0)),
                float(row.get("ema_20", price)),
                float(row.get("volatility", 0.0)),
                self.position_size,
                self._unrealized_pnl(price),
                self.time_in_trade,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):  # noqa: D401 - Gym API signature
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int):
        prev_value = self.portfolio_value
        self.last_action = int(action)

        price = self._current_price()
        if action == 0:
            self.position_size = min(1.0, self.position_size + 0.25)
            if self.position_size > 0 and self.time_in_trade == 0:
                self.entry_price = price
        elif action == 1:
            self.position_size = max(0.0, self.position_size - 0.25)
        elif action == 3:
            self.position_size = 0.0
            self.time_in_trade = 0.0

        self.step_idx += 1
        terminated = self.step_idx >= len(self.df) - 1 or self.step_idx >= self.cfg.max_steps
        truncated = False

        new_price = self._current_price()
        self.time_in_trade = self.time_in_trade + 1 if self.position_size > 0 else 0.0
        self.portfolio_value = self.cash + self._unrealized_pnl(new_price)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        pnl_change = self.portfolio_value - prev_value
        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1.0)
        drawdown_penalty = self.cfg.drawdown_penalty * drawdown
        overtrade_penalty = self.cfg.overtrade_penalty if action in (0, 1, 3) else 0.0
        reward = pnl_change - drawdown_penalty - overtrade_penalty

        return self._obs(), float(reward), terminated, truncated, {"portfolio_value": self.portfolio_value}

