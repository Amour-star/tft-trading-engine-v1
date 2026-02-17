"""
Executor factory and backward-compatible ExecutionEngine shim.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from config.settings import settings
from execution.base_executor import BaseExecutor
from execution.live_executor import LiveExecutor
from execution.paper_executor import PaperExecutor

if TYPE_CHECKING:
    from data.fetcher import KuCoinDataFetcher


def create_executor(fetcher: "KuCoinDataFetcher") -> BaseExecutor:
    mode = settings.trading.trading_mode.upper()
    if mode == "PAPER":
        return PaperExecutor(fetcher)
    if mode == "LIVE":
        return LiveExecutor(fetcher)
    raise ValueError(f"Unknown TRADING_MODE '{settings.trading.trading_mode}'")


class ExecutionEngine:
    """
    Backward-compatible constructor alias.
    Returns a mode-specific executor instance.
    """

    def __new__(cls, fetcher: "KuCoinDataFetcher") -> BaseExecutor:
        return create_executor(fetcher)


__all__ = [
    "BaseExecutor",
    "LiveExecutor",
    "PaperExecutor",
    "ExecutionEngine",
    "create_executor",
]
