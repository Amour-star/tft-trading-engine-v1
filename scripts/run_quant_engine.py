"""Run the quant hedge-fund style multi-asset paper trading engine."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from quant import QuantEngineConfig, QuantTradingOrchestrator
from utils.logging import setup_logging


async def _run() -> None:
    os.environ["QUANT_ENGINE_ENABLED"] = "true"
    cfg = QuantEngineConfig().normalized()
    setup_logging()
    logger.info(
        "Starting quant orchestrator | symbols={} | market={}s signal={}s rebalance={}s",
        cfg.universe,
        cfg.market_interval_sec,
        cfg.signal_interval_sec,
        cfg.rebalance_interval_sec,
    )
    engine = QuantTradingOrchestrator(cfg)
    await engine.run()


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
