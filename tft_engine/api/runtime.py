"""
Runtime service container for API and scripts.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from tft_engine.ai.llm.opus_service import OpusIntegrationService
from tft_engine.ai.model_registry import ModelRegistryService
from tft_engine.ai.rl.inference import RLInferenceService
from tft_engine.ai.supervised.inference import SupervisedInferenceService
from tft_engine.database.connection import get_session
from tft_engine.database.migrations import initialize_database
from tft_engine.engine.ai_controller import AIController
from tft_engine.engine.risk_manager import RiskManager
from tft_engine.engine.trade_executor import TradeExecutor

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_runtime():
    initialize_database()
    supervised = SupervisedInferenceService()
    rl = RLInferenceService()
    executor = TradeExecutor(get_session)
    risk = RiskManager()
    registry = ModelRegistryService(get_session)
    opus_service = OpusIntegrationService(get_session)
    controller = AIController(supervised, rl, executor, risk, registry)
    try:
        controller.reload_models_if_needed()
    except Exception:
        logger.warning("No registered models loaded at startup; using file defaults when available.")
    controller.start_model_watch()
    return {
        "ai_controller": controller,
        "executor": executor,
        "registry": registry,
        "opus_service": opus_service,
    }
