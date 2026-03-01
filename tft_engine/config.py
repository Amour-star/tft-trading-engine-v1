"""
Runtime configuration for TFT Trading Engine v2.

This module intentionally uses SQLite-only defaults and does not depend on
container-specific settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "supervised").mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "rl").mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EngineConfig:
    ai_enabled: bool = _env_bool("AI_ENABLED", True)
    min_confidence: float = _env_float("MIN_CONFIDENCE", 0.65)
    rl_enabled: bool = _env_bool("RL_ENABLED", True)
    model_check_interval: int = _env_int("MODEL_CHECK_INTERVAL", 300)
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///data/tft.db")
    opus_enabled: bool = _env_bool("OPUS_ENABLED", False)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_base_url: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages")
    opus_model: str = os.getenv("OPUS_MODEL", "claude-opus-4-1")
    llm_timeout_seconds: int = _env_int("LLM_TIMEOUT_SECONDS", 45)
    llm_cache_ttl_seconds: int = _env_int("LLM_CACHE_TTL_SECONDS", 300)
    llm_max_input_tokens: int = _env_int("LLM_MAX_INPUT_TOKENS", 4000)
    llm_max_output_tokens: int = _env_int("LLM_MAX_OUTPUT_TOKENS", 800)
    llm_daily_budget_usd: float = _env_float("LLM_DAILY_BUDGET_USD", 10.0)
    llm_monthly_budget_usd: float = _env_float("LLM_MONTHLY_BUDGET_USD", 200.0)
    llm_max_request_cost_usd: float = _env_float("LLM_MAX_REQUEST_COST_USD", 1.0)
    llm_cost_input_per_1k_tokens: float = _env_float("LLM_COST_INPUT_PER_1K_TOKENS", 0.015)
    llm_cost_output_per_1k_tokens: float = _env_float("LLM_COST_OUTPUT_PER_1K_TOKENS", 0.075)
    supervised_model_path: str = os.getenv(
        "SUPERVISED_MODEL_PATH",
        str((MODEL_DIR / "supervised" / "latest_xgb.pkl").as_posix()),
    )
    rl_model_path: str = os.getenv("RL_MODEL_PATH", str((MODEL_DIR / "rl" / "latest_rl.zip").as_posix()))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()


config = EngineConfig()

# Constants for compatibility with requested v2 interface.
AI_ENABLED = config.ai_enabled
MIN_CONFIDENCE = config.min_confidence
RL_ENABLED = config.rl_enabled
MODEL_CHECK_INTERVAL = config.model_check_interval
