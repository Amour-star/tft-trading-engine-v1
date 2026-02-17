"""
Fetch historical data for model training.
Downloads OHLCV data for top USDT pairs across multiple timeframes.
"""
import sys
sys.path.insert(0, ".")

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import settings
from data.fetcher import KuCoinDataFetcher
from data.features import compute_features
from utils.logging import setup_logging

DATA_DIR = Path("data/historical")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Fetch historical data")
    parser.add_argument("--pairs", type=int, default=10, help="Number of top pairs to fetch")
    parser.add_argument("--months", type=int, default=6, help="Months of history")
    parser.add_argument("--timeframes", nargs="+", default=["15min", "1hour"],
                        help="Timeframes to fetch")
    args = parser.parse_args()

    fetcher = KuCoinDataFetcher()

    # Get top pairs
    logger.info(f"Fetching top {args.pairs} USDT pairs...")
    top_pairs = fetcher.get_top_usdt_pairs(args.pairs)
    pair_symbols = [p["symbol"] for p in top_pairs]
    logger.info(f"Pairs: {pair_symbols}")

    # Always include BTC-USDT for correlation features
    if "BTC-USDT" not in pair_symbols:
        pair_symbols.append("BTC-USDT")

    # Fetch BTC data first (needed for features)
    logger.info("Fetching BTC-USDT data...")
    btc_dfs = {}
    for tf in args.timeframes:
        btc_df = fetcher.fetch_history("BTC-USDT", tf, args.months)
        btc_dfs[tf] = btc_df
        btc_df.to_parquet(DATA_DIR / f"BTC-USDT_{tf}.parquet", index=False)
        logger.info(f"BTC-USDT {tf}: {len(btc_df)} candles")

    # Fetch each pair
    for symbol in pair_symbols:
        logger.info(f"\nFetching {symbol}...")
        for tf in args.timeframes:
            df = fetcher.fetch_history(symbol, tf, args.months)
            if df.empty:
                logger.warning(f"No data for {symbol} {tf}")
                continue

            # Compute features
            btc_df = btc_dfs.get(tf, pd.DataFrame())
            df["pair"] = symbol
            df = compute_features(df, btc_df)

            # Save
            output_path = DATA_DIR / f"{symbol}_{tf}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"{symbol} {tf}: {len(df)} candles with {len(df.columns)} features → {output_path}")

    logger.info("\nData fetch complete!")
    logger.info(f"Files saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
