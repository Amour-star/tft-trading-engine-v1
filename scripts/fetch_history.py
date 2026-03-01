"""
Fetch historical data for model training.
Downloads OHLCV data for top USDT pairs across multiple timeframes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fetcher import KuCoinDataFetcher
from data.features import compute_features
from config.settings import XRP_ONLY_SYMBOL
from utils.logging import setup_logging

DATA_DIR = ROOT / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Fetch historical data")
    parser.add_argument("--pairs", type=int, default=10, help="Number of top pairs to fetch")
    parser.add_argument("--months", type=int, default=6, help="Months of history")
    parser.add_argument("--min-rows", type=int, default=220, help="Minimum rows required per file")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["15min", "1hour"],
        help="Timeframes to fetch",
    )
    args = parser.parse_args()

    fetcher = KuCoinDataFetcher()

    logger.info(f"Fetching top {args.pairs} USDT pairs...")
    top_pairs = fetcher.get_top_usdt_pairs(args.pairs)
    pair_symbols = [str(p.get("symbol", "")).strip() for p in top_pairs if str(p.get("symbol", "")).strip()]
    if not pair_symbols:
        pair_symbols = [XRP_ONLY_SYMBOL]
        logger.warning("Top pair discovery returned no symbols. Using fallback pairs.")

    # Always include XRP in the fetch universe so the trained model vocabulary can support XRP-only mode.
    pair_symbols = list(dict.fromkeys([XRP_ONLY_SYMBOL, *pair_symbols]))
    logger.info(f"Pairs: {pair_symbols}")

    logger.info(f"Fetching {XRP_ONLY_SYMBOL} reference data...")
    btc_dfs: dict[str, pd.DataFrame] = {}
    for tf in args.timeframes:
        btc_df = fetcher.fetch_history(XRP_ONLY_SYMBOL, tf, args.months)
        if btc_df.empty:
            logger.warning(f"{XRP_ONLY_SYMBOL} {tf} returned no data; using safe feature defaults.")
        else:
            btc_df.to_parquet(DATA_DIR / f"{XRP_ONLY_SYMBOL}_{tf}.parquet", index=False)
            logger.info(f"{XRP_ONLY_SYMBOL} {tf}: {len(btc_df)} candles")
        btc_dfs[tf] = btc_df

    written_files = 0
    for symbol in pair_symbols:
        logger.info(f"\nFetching {symbol}...")
        for tf in args.timeframes:
            df = fetcher.fetch_history(symbol, tf, args.months)
            if df.empty:
                if symbol == XRP_ONLY_SYMBOL:
                    raise RuntimeError(f"Required symbol {XRP_ONLY_SYMBOL} returned no data for timeframe {tf}")
                logger.warning(f"No data for {symbol} {tf}")
                continue
            if len(df) < args.min_rows:
                if symbol == XRP_ONLY_SYMBOL:
                    raise RuntimeError(
                        f"Required symbol {XRP_ONLY_SYMBOL} has only {len(df)} rows for {tf} "
                        f"(min required: {args.min_rows})"
                    )
                logger.warning(
                    f"Skipping {symbol} {tf}: only {len(df)} rows (< {args.min_rows})"
                )
                continue

            btc_df = btc_dfs.get(tf, pd.DataFrame())
            df["pair"] = symbol
            try:
                df = compute_features(df, btc_df)
            except Exception as exc:
                if symbol == XRP_ONLY_SYMBOL:
                    raise RuntimeError(
                        f"Feature engineering failed for required symbol {XRP_ONLY_SYMBOL} {tf}: {exc}"
                    ) from exc
                logger.error(f"Feature engineering failed for {symbol} {tf}: {exc}")
                continue

            output_path = DATA_DIR / f"{symbol}_{tf}.parquet"
            df.to_parquet(output_path, index=False)
            written_files += 1
            logger.info(
                f"{symbol} {tf}: {len(df)} candles with {len(df.columns)} features -> {output_path}"
            )

    logger.info("\nData fetch complete!")
    logger.info(f"Files saved to: {DATA_DIR}")

    if written_files == 0:
        raise RuntimeError("No historical parquet files were generated.")


if __name__ == "__main__":
    main()

