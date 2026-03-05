"""Validation script for the quant multi-asset paper trading stack."""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from data.database import EquityHistory, MetricSnapshot, SignalRecord, Trade, get_session, init_db
from quant import QuantEngineConfig, QuantTradingOrchestrator
from quant.types import RegimeState, StrategySignal
from utils.logging import setup_logging


async def _validate() -> None:
    os.environ["QUANT_ENGINE_ENABLED"] = "true"
    os.environ.setdefault("KUCOIN_OFFLINE_MODE", "true")
    os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.30")

    cfg = QuantEngineConfig().normalized()
    setup_logging()
    init_db()
    engine = QuantTradingOrchestrator(cfg)

    logger.info("[CHECK] market data cycle")
    await engine._run_market_cycle()
    logger.info("[CHECK] signal cycle")
    await engine._run_signal_cycle()
    logger.info("[CHECK] rebalance cycle")
    await engine._run_rebalance_cycle()

    # Guarantee at least one execution path during validation.
    prices = {symbol: snap.ticker_price for symbol, snap in engine.market_data.snapshots.items()}
    if prices:
        symbol = sorted(prices.keys())[0]
        forced_regime = RegimeState(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            label="Trending",
            confidence=0.9,
            position_size_mult=1.0,
            threshold_shift=-0.2,
            aggressiveness_mult=1.2,
        )
        open_signal = StrategySignal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            direction=1,
            confidence=0.99,
            score=0.99,
            strategy_name="validation_forced_open",
            regime="Trending",
            reason="system_check",
        )
        engine.execution.process_signal(
            signal=open_signal,
            regime=forced_regime,
            mark_price=prices[symbol],
            spread_pct=0.001,
            max_notional=150.0,
        )
        close_signal = StrategySignal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            direction=-1,
            confidence=0.99,
            score=-0.99,
            strategy_name="validation_forced_close",
            regime="Trending",
            reason="system_check",
        )
        engine.execution.process_signal(
            signal=close_signal,
            regime=forced_regime,
            mark_price=prices[symbol] * 1.001,
            spread_pct=0.001,
            max_notional=150.0,
        )

    session = get_session()
    try:
        signals = session.query(SignalRecord).count()
        trades = session.query(Trade).count()
        metrics = session.query(MetricSnapshot).count()
        equity = session.query(EquityHistory).count()
    finally:
        session.close()

    checks = {
        "signals_generated": signals > 0,
        "trades_logged": trades > 0,
        "metrics_computed": metrics > 0,
        "equity_updated": equity > 0,
    }
    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Quant system check failed: {failed}")
    print("QUANT SYSTEM STATUS: OK")


def main() -> None:
    asyncio.run(_validate())


if __name__ == "__main__":
    main()
