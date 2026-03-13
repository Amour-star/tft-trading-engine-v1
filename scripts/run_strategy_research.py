"""Run the automated strategy discovery and paper deployment loop."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from research.strategy_generator.loop import StrategyResearchLoop
from utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated strategy research loop")
    parser.add_argument("--symbol", default="XRP-USDT", help="Symbol to research")
    parser.add_argument("--timeframe", default="15min", help="Timeframe to research")
    parser.add_argument("--candidates", type=int, default=None, help="Number of random candidates")
    parser.add_argument("--top-percentile", type=float, default=None, help="Top percentile to deploy")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--sleep-seconds", type=float, default=None, help="Delay between continuous runs")
    args = parser.parse_args()

    try:
        setup_logging()
    except PermissionError:
        logger.remove()
        logger.add(sys.stdout, level="INFO")
    logger.info(
        "Starting strategy research | symbol={} timeframe={} continuous={}",
        args.symbol,
        args.timeframe,
        args.continuous,
    )
    loop = StrategyResearchLoop()
    if args.continuous:
        loop.run_forever(
            symbol=args.symbol,
            timeframe=args.timeframe,
            sleep_seconds=args.sleep_seconds,
        )
        return

    summary = loop.run_once(
        symbol=args.symbol,
        timeframe=args.timeframe,
        candidate_count=args.candidates,
        top_percentile=args.top_percentile,
    )
    logger.info(
        "Strategy research summary | run_id={} deployed={} accepted={}",
        summary.run_id,
        summary.deployed_count,
        summary.accepted_count,
    )


if __name__ == "__main__":
    main()
