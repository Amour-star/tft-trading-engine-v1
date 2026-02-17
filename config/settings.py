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


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in ("true", "1", "yes")


def _env_float(key: str, default: float = 0.0) -> float:
    return float(_env(key, str(default)))


def _env_int(key: str, default: int = 0) -> int:
    return int(_env(key, str(default)))


def _resolve_trading_mode() -> str:
    mode = os.getenv("TRADING_MODE")
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
    host: str = _env("POSTGRES_HOST", "localhost")
    port: int = _env_int("POSTGRES_PORT", 5432)
    db: str = _env("POSTGRES_DB", "tft_trading")
    user: str = _env("POSTGRES_USER", "trader")
    password: str = _env("POSTGRES_PASSWORD", "")

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class RedisConfig:
    host: str = _env("REDIS_HOST", "localhost")
    port: int = _env_int("REDIS_PORT", 6379)


@dataclass(frozen=True)
class TradingConfig:
    risk_per_trade: float = _env_float("RISK_PER_TRADE", 0.01)
    confidence_threshold: float = _env_float("CONFIDENCE_THRESHOLD", 0.55)
    max_daily_loss_pct: float = _env_float("MAX_DAILY_LOSS_PCT", 0.03)
    max_consecutive_losses: int = _env_int("MAX_CONSECUTIVE_LOSSES", 5)
    trading_enabled: bool = _env_bool("TRADING_ENABLED", True)
    trading_mode: str = _resolve_trading_mode()
    allow_live_trading: str = _env("ALLOW_LIVE_TRADING", "")
    paper_starting_balance: float = _env_float("PAPER_STARTING_BALANCE", 10_000.0)
    paper_fee_rate: float = _env_float("PAPER_FEE_RATE", 0.001)
    paper_slippage_bps: float = _env_float("PAPER_SLIPPAGE_BPS", 0.0)
    paper_db_path: str = _env("PAPER_DB_PATH", "/data/paper_trading.db")
    max_open_trades: int = 1
    top_pairs_count: int = 30
    quote_currency: str = "USDT"
    min_volume_24h: float = 500_000.0

    def __post_init__(self) -> None:
        mode = self.trading_mode.strip().upper()
        if mode not in {"PAPER", "LIVE"}:
            raise ValueError(
                f"Unknown TRADING_MODE '{self.trading_mode}'. Expected PAPER or LIVE."
            )
        if mode == "LIVE" and self.allow_live_trading != "YES_I_UNDERSTAND":
            raise ValueError(
                "LIVE mode blocked. Set ALLOW_LIVE_TRADING=YES_I_UNDERSTAND to enable."
            )

    @property
    def paper_trading(self) -> bool:
        return self.trading_mode.upper() == "PAPER"


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
    api_port: int = _env_int("API_PORT", 8000)
    dashboard_port: int = _env_int("DASHBOARD_PORT", 8501)
    password: str = _env("DASHBOARD_PASSWORD", "admin")


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    kucoin: KuCoinConfig = field(default_factory=KuCoinConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    log_level: str = _env("LOG_LEVEL", "INFO")
    log_dir: Path = Path(_env("LOG_DIR", str(BASE_DIR / "logs")))


settings = Settings()
