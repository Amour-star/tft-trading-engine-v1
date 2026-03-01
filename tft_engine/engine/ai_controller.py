"""
Hybrid AI controller.

Decision layer:
- Supervised model decides IF a trade should be opened.
- RL agent decides HOW the trade should be managed.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np

from tft_engine.ai.model_registry import ModelRegistryService
from tft_engine.ai.rl.inference import RLInferenceService
from tft_engine.ai.supervised.inference import SupervisedInferenceService
from tft_engine.config import config
from tft_engine.engine.risk_manager import RiskManager
from tft_engine.engine.trade_executor import TradeExecutor

logger = logging.getLogger(__name__)


class AIController:
    def __init__(
        self,
        supervised_service: SupervisedInferenceService,
        rl_service: RLInferenceService,
        trade_executor: TradeExecutor,
        risk_manager: RiskManager,
        registry: ModelRegistryService,
    ) -> None:
        self.supervised = supervised_service
        self.rl = rl_service
        self.executor = trade_executor
        self.risk = risk_manager
        self.registry = registry
        self.min_confidence = config.min_confidence
        self.last_prediction: Optional[dict[str, Any]] = None
        self._model_versions = {"xgboost": None, "rl": None}
        self._stop = threading.Event()
        self._reload_thread: Optional[threading.Thread] = None

    def start_model_watch(self) -> None:
        if self._reload_thread and self._reload_thread.is_alive():
            return
        self._reload_thread = threading.Thread(target=self._watch_models_loop, daemon=True)
        self._reload_thread.start()
        logger.info("Started model auto-reload watcher.")

    def stop_model_watch(self) -> None:
        self._stop.set()
        if self._reload_thread and self._reload_thread.is_alive():
            self._reload_thread.join(timeout=2)

    def _watch_models_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.reload_models_if_needed()
            except Exception:
                logger.exception("Model reload watcher failed.")
            self._stop.wait(config.model_check_interval)

    def reload_models_if_needed(self) -> None:
        latest_xgb = self.registry.latest("xgboost")
        if latest_xgb and latest_xgb.version != self._model_versions["xgboost"]:
            self.supervised = SupervisedInferenceService(model_path=latest_xgb.path)
            self.supervised.load_model()
            self._model_versions["xgboost"] = latest_xgb.version
            logger.info(f"Reloaded xgboost model version={latest_xgb.version}")

        latest_rl = self.registry.latest("rl")
        if latest_rl and latest_rl.version != self._model_versions["rl"]:
            self.rl = RLInferenceService(model_path=latest_rl.path)
            self.rl.load_model()
            self._model_versions["rl"] = latest_rl.version
            logger.info(f"Reloaded RL model version={latest_rl.version}")

    def _rl_observation(self, market_snapshot: dict, quantity: float) -> np.ndarray:
        return np.asarray(
            [
                float(market_snapshot.get("current_price", market_snapshot.get("entry_price", 0.0))),
                float(market_snapshot.get("rsi", 50.0)),
                float(market_snapshot.get("ema_20", market_snapshot.get("current_price", 0.0))),
                float(market_snapshot.get("volatility", 0.0)),
                float(quantity),
                float(market_snapshot.get("unrealized_pnl", 0.0)),
                float(market_snapshot.get("time_in_trade", 0.0)),
            ],
            dtype=np.float32,
        )

    def _action_to_size_multiplier(self, action: int) -> float:
        mapping = {
            0: 1.20,  # increase
            1: 0.75,  # decrease
            2: 1.00,  # hold
            3: 0.00,  # close / no entry
        }
        return mapping.get(int(action), 1.0)

    def evaluate_and_open_trade(
        self,
        market_snapshot: dict,
        account_balance: float,
        side: str = "BUY",
        fees: float = 0.0,
    ) -> dict[str, Any]:
        if not config.ai_enabled:
            return {"opened": False, "reason": "AI disabled by configuration."}

        prediction = self.supervised.predict(market_snapshot)
        self.last_prediction = {
            "confidence": prediction.confidence_score,
            "win_probability": prediction.win_probability,
            "expected_return": prediction.expected_return,
            "model_version": prediction.model_version,
            "timestamp": int(time.time()),
        }

        if prediction.win_probability < self.min_confidence:
            return {
                "opened": False,
                "reason": "Rejected by confidence threshold.",
                "win_probability": prediction.win_probability,
                "confidence_score": prediction.confidence_score,
            }

        base_qty = self.risk.position_size(
            balance=account_balance,
            confidence=prediction.win_probability,
            volatility=float(market_snapshot.get("volatility", 0.0)),
            entry_price=float(market_snapshot.get("entry_price", market_snapshot.get("current_price", 0.0))),
        )

        if base_qty <= 0:
            return {"opened": False, "reason": "Calculated quantity is zero."}

        if config.rl_enabled:
            rl_obs = self._rl_observation(market_snapshot, quantity=base_qty)
            action = self.rl.choose_action(rl_obs)
            quantity = base_qty * self._action_to_size_multiplier(action)
            if action == 3 or quantity <= 0:
                return {"opened": False, "reason": "RL advised no entry / immediate close.", "rl_action": action}
        else:
            action = 2
            quantity = base_qty

        trade = self.executor.open_trade(
            symbol=str(market_snapshot["symbol"]),
            side=side,
            entry_price=float(market_snapshot.get("entry_price", market_snapshot.get("current_price", 0.0))),
            quantity=quantity,
            fees=fees,
            ai_confidence=prediction.confidence_score,
            ai_score=prediction.win_probability,
            model_version=prediction.model_version,
            feature_payload={
                "rsi": market_snapshot.get("rsi"),
                "ema_20": market_snapshot.get("ema_20"),
                "ema_50": market_snapshot.get("ema_50"),
                "atr": market_snapshot.get("atr"),
                "volatility": market_snapshot.get("volatility"),
                "volume": market_snapshot.get("volume"),
                "macd": market_snapshot.get("macd"),
                "market_regime": market_snapshot.get("market_regime"),
            },
        )
        return {
            "opened": True,
            "trade_id": trade.id,
            "quantity": quantity,
            "win_probability": prediction.win_probability,
            "confidence_score": prediction.confidence_score,
            "model_version": prediction.model_version,
            "rl_action": action,
        }

    def manage_open_trade(self, trade_snapshot: dict) -> dict[str, Any]:
        """
        RL action policy for position sizing, stop/take adaptation, and exit timing.
        """
        if not config.rl_enabled:
            return {"action": "hold"}

        obs = self._rl_observation(
            {
                "current_price": trade_snapshot.get("current_price"),
                "rsi": trade_snapshot.get("rsi", 50.0),
                "ema_20": trade_snapshot.get("ema_20", trade_snapshot.get("current_price", 0.0)),
                "volatility": trade_snapshot.get("volatility", 0.0),
                "unrealized_pnl": trade_snapshot.get("unrealized_pnl", 0.0),
                "time_in_trade": trade_snapshot.get("time_in_trade", 0.0),
            },
            quantity=float(trade_snapshot.get("quantity", 0.0)),
        )
        action = self.rl.choose_action(obs)
        action_name = {0: "increase", 1: "decrease", 2: "hold", 3: "close"}.get(action, "hold")

        atr = float(trade_snapshot.get("atr", 0.0))
        entry_price = float(trade_snapshot.get("entry_price", 0.0))
        side = str(trade_snapshot.get("side", "BUY"))
        stop = self.risk.stop_loss_price(side, entry_price, atr) if atr > 0 else None
        take = self.risk.take_profit_price(side, entry_price, stop) if stop is not None else None
        return {
            "action": action_name,
            "rl_action": action,
            "suggested_stop_loss": stop,
            "suggested_take_profit": take,
        }
