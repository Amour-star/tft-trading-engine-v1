"""
Shared test fixtures: in-memory SQLite database, mock fetcher, mock predictor.
"""
import os
import sys

# Force paper trading mode for all tests
os.environ["TRADING_MODE"] = "PAPER"
os.environ["POSTGRES_HOST"] = ""
os.environ["POSTGRES_PASSWORD"] = ""

# Relax safety caps in unit tests unless a test explicitly overrides them.
os.environ.setdefault("MAX_POSITION_PCT", "1.0")
os.environ.setdefault("MAX_NOTIONAL_PER_TRADE", "1000000000000")
os.environ.setdefault("KILL_SWITCH_DAILY_DRAWDOWN", "1.0")
os.environ.setdefault("KILL_SWITCH_BALANCE_DROP", "1.0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from data.database import Base


# All modules that import get_session at module level
_GET_SESSION_TARGETS = [
    "data.database",
    "data",
    "execution.monitor",
    "execution.base_executor",
    "execution.live_executor",
    "engine.safety",
    "engine.feedback",
    "engine.main",
    "engine.attribution",
    "engine.performance_metrics",
    "engine.strategy_evolution",
    "engine.governance",
    "dashboard.api",
    "risk.prop_risk_manager",
    "paper.reset",
    "services.reconciliation",
]


# ---------------------------------------------------------------------------
# Database fixtures (in-memory SQLite with StaticPool for thread-safety)
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_engine():
    """
    Create an in-memory SQLite database engine with all tables.
    Uses StaticPool + check_same_thread=False so that all threads
    (including FastAPI's run_in_threadpool) share the same connection.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture()
def db_session(db_engine):
    """Create a session factory bound to the in-memory database."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture()
def patch_db(db_engine):
    """
    Patch get_session in all modules to use the in-memory SQLite database.
    """
    Session = sessionmaker(bind=db_engine)

    def _get_session():
        return Session()

    patches = []
    for target in _GET_SESSION_TARGETS:
        try:
            p = patch(f"{target}.get_session", _get_session)
            p.start()
            patches.append(p)
        except AttributeError:
            pass
        try:
            p = patch(f"{target}.get_engine", lambda: db_engine)
            p.start()
            patches.append(p)
        except AttributeError:
            pass

    engine_patch = patch("data.database.get_engine", lambda: db_engine)
    engine_patch.start()
    patches.append(engine_patch)

    yield _get_session

    for p in patches:
        p.stop()


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 300, base_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    np.random.seed(seed)
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=15 * i) for i in range(n)]
    close = base_price + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.exponential(1000, n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_mock_fetcher(current_price: float = 100.0) -> MagicMock:
    """Create a mock KuCoinDataFetcher."""
    fetcher = MagicMock()
    fetcher.get_ticker.return_value = {
        "price": current_price,
        "best_bid": current_price - 0.01,
        "best_ask": current_price + 0.01,
        "size": 1.0,
        "time": int(datetime.utcnow().timestamp() * 1000),
    }
    fetcher.get_spread.return_value = (0.02, 0.0002)
    fetcher.get_symbol_info.return_value = {
        "symbol": "XRP-USDT",
        "base_currency": "XRP",
        "quote_currency": "USDT",
        "base_min_size": 0.00001,
        "base_max_size": 10000.0,
        "base_increment": 0.00001,
        "price_increment": 0.01,
        "quote_min_size": 0.01,
        "quote_increment": 0.01,
        "fee_currency": "USDT",
    }
    fetcher.get_top_usdt_pairs.return_value = [
        {"symbol": "XRP-USDT", "volValue": "180000000"},
    ]
    fetcher.fetch_klines.return_value = make_ohlcv(300, base_price=current_price)
    fetcher.get_balance.return_value = 10000.0
    return fetcher


def make_mock_predictor(
    prob_up: float = 0.7,
    prob_down: float = 0.3,
    expected_move: float = 0.02,
    confidence: float = 0.75,
) -> MagicMock:
    """Create a mock TFTPredictor."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "prob_up": prob_up,
        "prob_down": prob_down,
        "expected_move": expected_move,
        "confidence": confidence,
        "forecast_vector": [0.01, 0.015, 0.02, 0.018, 0.022],
        "lower_bound": [-0.01, -0.005, 0.0, 0.005, 0.01],
        "upper_bound": [0.03, 0.035, 0.04, 0.035, 0.04],
        "model_version": "test_v1",
    }
    predictor.model = True  # Non-None to pass model checks
    predictor.model_version = "test_v1"
    predictor.get_supported_pairs.return_value = ["XRP-USDT"]
    return predictor


@pytest.fixture()
def mock_fetcher():
    return make_mock_fetcher()


@pytest.fixture()
def mock_predictor():
    return make_mock_predictor()
