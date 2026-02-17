"""
Run backtesting with walk-forward validation and Monte Carlo simulation.
Determines if the system is safe to go live.
"""
import sys
sys.path.insert(0, ".")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from backtesting.backtester import Backtester
from utils.logging import setup_logging

DATA_DIR = Path("data/historical")


def generate_mock_signals(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Generate trading signals based on simple momentum + mean reversion.
    In production, this would use the actual TFT model predictions.
    """
    signals = []
    if test_df.empty or len(test_df) < 20:
        return signals

    for i in range(20, len(test_df), 10):  # Signal every ~10 candles
        row = test_df.iloc[i]

        # Simple signal logic (placeholder for TFT predictions)
        rsi = row.get("rsi_14", 50)
        momentum = row.get("momentum_10", 0)
        vol_regime = row.get("volatility_regime", "normal")

        if vol_regime == "extreme":
            continue

        confidence = 0.5
        if rsi < 35 and momentum < -0.02:
            confidence = 0.75  # Oversold bounce
        elif rsi < 45 and momentum > 0:
            confidence = 0.70  # Trend continuation
        else:
            continue

        entry = float(row["close"])
        atr = float(row.get("atr_14", entry * 0.01))
        stop = entry - atr * 2
        target = entry + atr * 3

        signals.append({
            "timestamp": row["timestamp"],
            "pair": row.get("pair", "UNKNOWN"),
            "side": "BUY",
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "confidence": confidence,
        })

    return signals


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--pair", default="BTC-USDT", help="Pair to backtest")
    parser.add_argument("--timeframe", default="15min", help="Timeframe")
    parser.add_argument("--balance", type=float, default=10000, help="Starting balance")
    parser.add_argument("--monte-carlo", type=int, default=1000, help="Monte Carlo simulations")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BACKTEST ENGINE")
    logger.info("=" * 60)

    # Load data
    data_file = DATA_DIR / f"{args.pair}_{args.timeframe}.parquet"
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Run scripts/fetch_history.py first")
        return

    df = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(df)} candles for {args.pair} {args.timeframe}")

    # Initialize backtester
    bt = Backtester(initial_balance=args.balance)

    # Walk-forward backtest
    logger.info("\n--- Walk-Forward Validation ---")
    wf_result = bt.walk_forward(
        df=df,
        signal_generator=generate_mock_signals,
        window_size=2000,
        step_size=500,
        risk_per_trade=settings.trading.risk_per_trade,
    )

    logger.info(f"\nWalk-Forward Results:")
    logger.info(f"  Trades: {wf_result.total_trades}")
    logger.info(f"  Win Rate: {wf_result.win_rate:.2%}")
    logger.info(f"  Sharpe Ratio: {wf_result.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: {wf_result.max_drawdown_pct:.2%}")
    logger.info(f"  Total PnL: ${wf_result.total_pnl:.2f}")
    logger.info(f"  Avg R: {wf_result.avg_r_multiple:.2f}")
    logger.info(f"  Profit Factor: {wf_result.profit_factor:.2f}")
    logger.info(f"  PASSED: {wf_result.passed}")
    if wf_result.failure_reasons:
        for r in wf_result.failure_reasons:
            logger.warning(f"  FAIL: {r}")

    # Monte Carlo
    if wf_result.trades:
        logger.info(f"\n--- Monte Carlo ({args.monte_carlo} simulations) ---")
        mc_result = bt.monte_carlo(wf_result.trades, args.monte_carlo)

        logger.info(f"\nMonte Carlo Results:")
        logger.info(f"  Median Final Balance: ${mc_result['median_final_balance']:.2f}")
        logger.info(f"  P5 Balance: ${mc_result['p5_final_balance']:.2f}")
        logger.info(f"  P95 Balance: ${mc_result['p95_final_balance']:.2f}")
        logger.info(f"  Median Sharpe: {mc_result['median_sharpe']:.2f}")
        logger.info(f"  P5 Sharpe: {mc_result['p5_sharpe']:.2f}")
        logger.info(f"  P95 Max Drawdown: {mc_result['p95_max_drawdown']:.2%}")
        logger.info(f"  Prob Profit: {mc_result['probability_profit']:.2%}")
        logger.info(f"  Prob Ruin: {mc_result['probability_ruin']:.2%}")
        logger.info(f"  PASSED: {mc_result['passed']}")

    # Final verdict
    all_passed = wf_result.passed and (not wf_result.trades or mc_result.get("passed", False))
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✅ BACKTEST PASSED - System approved for live trading")
    else:
        logger.warning("❌ BACKTEST FAILED - DO NOT proceed to live trading")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
