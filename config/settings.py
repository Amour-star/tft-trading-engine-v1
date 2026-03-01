"""
Central configuration for the TFT Trading Engine.
All settings can be overridden via environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
XRP_ONLY_SYMBOL = "XRP-USDT"
TRADING_UNIVERSE: List[str] = [XRP_ONLY_SYMBOL]


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    value = _env(key, "").strip()
    if value == "":
        return default
    return value.lower() in ("true", "1", "yes")


def _env_float(key: str, default: float = 0.0) -> float:
    value = _env(key, "").strip()
    if value == "":
        return default
    return float(value)


def _env_int(key: str, default: int = 0) -> int:
    value = _env(key, "").strip()
    if value == "":
        return default
    return int(value)


def _resolve_trading_mode() -> str:
    mode = os.getenv("TRADING_MODE") or os.getenv("MODE")
    if mode:
        return mode.strip().upper()

    # Backward compatibility with legacy PAPER_TRADING flag.
    legacy = os.getenv("PAPER_TRADING")
    if legacy is not None:
        return "PAPER" if legacy.lower() in ("true", "1", "yes") else "LIVE"

    return "PAPER"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KuCoinConfig:
    api_key: str = _env("KUCOIN_API_KEY")
    api_secret: str = _env("KUCOIN_API_SECRET")
    api_passphrase: str = _env("KUCOIN_API_PASSPHRASE")
    sandbox: bool = _env_bool("KUCOIN_SANDBOX", False)

    @property
    def base_url(self) -> str:
        if self.sandbox:
            return "https://openapi-sandbox.kucoin.com"
        return "https://api.kucoin.com"


@dataclass(frozen=True)
class DatabaseConfig:
    database_mode: str = _env("DATABASE_MODE", "SQLITE").strip().upper()
    sqlite_path: str = _env("SQLITE_PATH", str(BASE_DIR / "data" / "tft_engine.db"))
    sqlite_wal_mode: bool = _env_bool("SQLITE_WAL_MODE", True)
    host: str = _env("POSTGRES_HOST", "localhost")
    port: int = _env_int("POSTGRES_PORT", 5432)
    db: str = _env("POSTGRES_DB", "tft_trading")
    user: str = _env("POSTGRES_USER", "trader")
    password: str = _env("POSTGRES_PASSWORD", "")
    pool_size: int = _env_int("POSTGRES_POOL_SIZE", 10)
    max_overflow: int = _env_int("POSTGRES_MAX_OVERFLOW", 20)

    def __post_init__(self) -> None:
        if self.database_mode not in {"SQLITE", "POSTGRES"}:
            raise ValueError(
                f"Unknown DATABASE_MODE '{self.database_mode}'. Expected SQLITE or POSTGRES."
            )

    @property
    def sqlite_resolved_path(self) -> Path:
        path = Path(self.sqlite_path)
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sqlite_url(self) -> str:
        return f"sqlite:///{self.sqlite_resolved_path.as_posix()}"

    @property
    def postgres_url(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class RedisConfig:
    host: str = _env("REDIS_HOST", "localhost")
    port: int = _env_int("REDIS_PORT", 6379)


@dataclass(frozen=True)
class TradingConfig:
    risk_per_trade: float = _env_float("RISK_PER_TRADE", 0.01)
    confidence_threshold: float = _env_float("CONFIDENCE_THRESHOLD", 0.50)
    aggression_level: float = _env_float("AGGRESSION_LEVEL", 1.0)
    allow_shorts: bool = _env_bool("ALLOW_SHORTS", True)
    max_daily_loss_pct: float = _env_float("MAX_DAILY_LOSS_PCT", 0.03)
    max_consecutive_losses: int = _env_int("MAX_CONSECUTIVE_LOSSES", 5)
    trading_enabled: bool = _env_bool("TRADING_ENABLED", True)
    trading_mode: str = _resolve_trading_mode()
    allow_live_trading: str = _env("ALLOW_LIVE_TRADING", "")
    paper_starting_balance: float = _env_float("PAPER_STARTING_BALANCE", 10_000.0)
    paper_fee_rate: float = _env_float("PAPER_FEE_RATE", 0.001)
    paper_slippage_bps: float = _env_float("PAPER_SLIPPAGE_BPS", 0.0)
    paper_db_path: str = _env("PAPER_DB_PATH", str(BASE_DIR / "data" / "paper_trading.db"))
    paper_require_live_price: bool = _env_bool("PAPER_REQUIRE_LIVE_PRICE", True)
    spot_only_mode: bool = _env_bool("SPOT_ONLY_MODE", True)
    max_open_trades: int = _env_int("MAX_OPEN_TRADES", 3)
    max_spread_pct: float = _env_float("MAX_SPREAD_PCT", 0.005)
    top_pairs_count: int = 1
    quote_currency: str = "USDT"
    min_volume_24h: float = 500_000.0

    def __post_init__(self) -> None:
        mode = self.trading_mode.strip().upper()
        if mode not in {"PAPER", "LIVE"}:
            raise ValueError(
                f"Unknown TRADING_MODE '{self.trading_mode}'. Expected PAPER or LIVE."
            )
        if mode == "LIVE" and self.allow_live_trading.strip().lower() != "true":
            raise RuntimeError(
                "LIVE mode blocked. Set ALLOW_LIVE_TRADING=true to enable."
            )
        if self.aggression_level <= 0:
            raise ValueError("AGGRESSION_LEVEL must be > 0")
        if self.max_open_trades <= 0:
            raise ValueError("MAX_OPEN_TRADES must be >= 1")
        if self.max_spread_pct <= 0:
            raise ValueError("MAX_SPREAD_PCT must be > 0")

    @property
    def paper_trading(self) -> bool:
        return self.trading_mode.upper() == "PAPER"

    @property
    def paper_initial_balance(self) -> float:
        return _env_float("PAPER_INITIAL_BALANCE", self.paper_starting_balance)


@dataclass(frozen=True)
class ModelConfig:
    timeframes: List[str] = field(default_factory=lambda: ["1min", "5min", "15min", "1hour"])
    history_months: int = 6
    forecast_horizons: int = 12
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 50
    encoder_length: int = 96
    prediction_length: int = 12
    retrain_interval_days: int = 7
    validation_weeks: int = 2
    auto_train_if_missing: bool = _env_bool("AUTO_TRAIN_IF_MISSING", False)
    auto_fetch_history_if_missing: bool = _env_bool("AUTO_FETCH_HISTORY_IF_MISSING", True)
    bootstrap_pairs: int = _env_int("BOOTSTRAP_PAIRS", 1)
    bootstrap_timeframe: str = _env("BOOTSTRAP_TIMEFRAME", "15min")
    bootstrap_history_months: int = _env_int("BOOTSTRAP_HISTORY_MONTHS", 6)


@dataclass(frozen=True)
class BacktestConfig:
    min_sharpe: float = 1.5
    max_drawdown_pct: float = 0.15
    slippage_bps: float = 5.0
    commission_bps: float = 10.0
    latency_ms: int = 50
    monte_carlo_runs: int = 1000


@dataclass(frozen=True)
class SafetyConfig:
    max_daily_loss_pct: float = _env_float("MAX_DAILY_LOSS_PCT", 0.03)
    max_consecutive_losses: int = _env_int("MAX_CONSECUTIVE_LOSSES", 5)
    circuit_breaker_volatility_multiplier: float = 3.0
    api_error_threshold: int = 5
    position_reconciliation_on_start: bool = True


@dataclass(frozen=True)
class DashboardConfig:
    api_host: str = _env("API_HOST", "127.0.0.1")
    api_port: int = _env_int("API_PORT", 8000)
    dashboard_host: str = _env("DASHBOARD_HOST", "127.0.0.1")
    dashboard_port: int = _env_int("DASHBOARD_PORT", 8501)
    cors_origins: str = _env("CORS_ORIGINS", "*")
    password: str = _env("DASHBOARD_PASSWORD", "admin")
    admin_token: str = _env("ADMIN_TOKEN", "change_me")


@dataclass(frozen=True)
class RuntimeConfig:
    environment: str = _env("ENVIRONMENT", _env("ENV", "development")).strip().lower()
    ai_enabled: bool = _env_bool("AI_ENABLED", True)
    ai_auditor_enabled: bool = _env_bool("AI_AUDITOR_ENABLED", False)


@dataclass(frozen=True)
class GovernanceConfig:
    llm_enabled: bool = _env_bool("LLM_ENABLED", False)
    provider: str = _env("LLM_PROVIDER", "anthropic")
    model: str = _env("OPUS_MODEL", "claude-opus-4-1")
    api_key: str = _env("ANTHROPIC_API_KEY", "")
    api_url: str = _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages")
    timeout_seconds: float = _env_float("LLM_TIMEOUT_SECONDS", 2.0)
    temperature: float = _env_float("LLM_TEMPERATURE", 0.1)
    max_output_tokens: int = _env_int("LLM_MAX_OUTPUT_TOKENS", 120)
    per_trade_cost_cap: float = _env_float("PER_TRADE_COST_CAP", 0.10)
    daily_cost_cap: float = _env_float("DAILY_COST_CAP", 5.00)
    max_failures_before_disable: int = _env_int("LLM_MAX_FAILURES", 3)
    cache_ttl_seconds: int = _env_int("LLM_CACHE_TTL_SECONDS", 300)
    input_cost_per_1k_tokens: float = _env_float("LLM_COST_INPUT_PER_1K_TOKENS", 0.015)
    output_cost_per_1k_tokens: float = _env_float("LLM_COST_OUTPUT_PER_1K_TOKENS", 0.075)


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    kucoin: KuCoinConfig = field(default_factory=KuCoinConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    log_level: str = _env("LOG_LEVEL", "INFO")
    log_dir: Path = Path(_env("LOG_DIR", str(BASE_DIR / "logs")))


settings = Settings()
