"""
CLI script to train PPO RL model from SQLite trade history.
"""
from __future__ import annotations

import logging
import sys

from tft_engine.ai.model_registry import ModelRegistryService
from tft_engine.ai.rl.train_rl import train_rl_from_database
from tft_engine.database.connection import get_session
from tft_engine.database.migrations import initialize_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_rl")


def main() -> int:
    try:
        initialize_database()
        metrics = train_rl_from_database(get_session)
        registry = ModelRegistryService(get_session)
        registry.register_model(
            model_type="rl",
            version=metrics["version"],
            path="models/rl/latest_rl.zip",
            metrics=metrics,
        )
        logger.info(f"RL training complete: {metrics}")
        return 0
    except Exception:
        logger.exception("RL training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
