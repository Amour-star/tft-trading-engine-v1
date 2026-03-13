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
from data.database import init_db, replace_historical_candles
import os

from config.settings import REFERENCE_SYMBOL, XRP_ONLY_SYMBOL
from utils.logging import setup_logging

DATA_DIR = ROOT / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _env_symbols() -> list[str]:
    raw = os.getenv("MARKET_DATA_SYMBOLS", "").strip()
    if not raw:
        return []
    parsed: list[str] = []
    for token in raw.split(","):
        symbol = token.strip().upper().replace("/", "-")
        if not symbol:
            continue
        if "-" not in symbol:
            symbol = f"{symbol}-USDT"
        parsed.append(symbol)
    return list(dict.fromkeys(parsed))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Fetch historical data")
    parser.add_argument("--pairs", type=int, default=10, help="Number of top pairs to fetch")
    parser.add_argument("--months", type=int, default=6, help="Months of history")
    parser.add_argument("--min-rows", type=int, default=220, help="Minimum rows required per file")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Explicit symbol universe to fetch (example: BTC-USDT ETH-USDT XRP-USDT DOGE-USDT)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["15min", "1hour"],
        help="Timeframes to fetch",
    )
    parser.add_argument(
        "--skip-db-store",
        action="store_true",
        help="Skip storing raw OHLCV history in the database",
    )
    args = parser.parse_args()

    fetcher = KuCoinDataFetcher()
    if not args.skip_db_store:
        init_db()

    pair_symbols = [str(symbol).strip().upper() for symbol in (args.symbols or []) if str(symbol).strip()]
    if not pair_symbols:
        pair_symbols = _env_symbols()
    if not pair_symbols:
        logger.info(f"Fetching top {args.pairs} USDT pairs...")
        top_pairs = fetcher.get_top_usdt_pairs(args.pairs)
        pair_symbols = [
            str(p.get("symbol", "")).strip()
            for p in top_pairs
            if str(p.get("symbol", "")).strip()
        ]
    if not pair_symbols:
        pair_symbols = [REFERENCE_SYMBOL, XRP_ONLY_SYMBOL]
        logger.warning("Symbol discovery returned no symbols. Using fallback pairs.")

    # Always include the benchmark reference pair and the active symbol.
    pair_symbols = list(dict.fromkeys([REFERENCE_SYMBOL, XRP_ONLY_SYMBOL, *pair_symbols]))
    logger.info(f"Pairs: {pair_symbols}")

    logger.info(f"Fetching {REFERENCE_SYMBOL} reference data...")
    reference_dfs: dict[str, pd.DataFrame] = {}
    for tf in args.timeframes:
        ref_df = fetcher.fetch_history(REFERENCE_SYMBOL, tf, args.months)
        if ref_df.empty:
            logger.warning(f"{REFERENCE_SYMBOL} {tf} returned no data; using safe feature defaults.")
        else:
            if not args.skip_db_store:
                stored_rows = replace_historical_candles(REFERENCE_SYMBOL, tf, ref_df)
                logger.info(f"{REFERENCE_SYMBOL} {tf}: stored {stored_rows} candles in database")
            ref_df.to_parquet(DATA_DIR / f"{REFERENCE_SYMBOL}_{tf}.parquet", index=False)
            logger.info(f"{REFERENCE_SYMBOL} {tf}: {len(ref_df)} candles")
        reference_dfs[tf] = ref_df

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

            if not args.skip_db_store:
                stored_rows = replace_historical_candles(symbol, tf, df)
                logger.info(f"{symbol} {tf}: stored {stored_rows} candles in database")

            ref_df = reference_dfs.get(tf, pd.DataFrame())
            df["pair"] = symbol
            try:
                df = compute_features(df, df if symbol == REFERENCE_SYMBOL else ref_df)
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

