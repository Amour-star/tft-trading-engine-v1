"""
Test the complete trading flow end-to-end without executing real orders.
Run this to diagnose where signals are being blocked.

Usage:
    python scripts/test_trading_flow.py
    python scripts/test_trading_flow.py --pair BTC-USDT
    python scripts/test_trading_flow.py --all-pairs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta
from loguru import logger

from config.settings import settings
from data.fetcher import KuCoinDataFetcher
from data.features import compute_features
from models.tft_model import TFTPredictor


def test_single_pair(
    fetcher: KuCoinDataFetcher,
    predictor: TFTPredictor,
    pair: str,
    btc_df=None,
) -> dict:
    """Test the full pipeline for a single pair and return diagnostics."""
    result = {"pair": pair, "stage": "start", "passed": False}

    # 1. Fetch data
    try:
        df = fetcher.fetch_klines(
            pair, "15min",
            start_dt=datetime.utcnow() - timedelta(hours=80),
        )
        result["rows"] = len(df) if not df.empty else 0
        if df.empty or len(df) < 200:
            result["stage"] = "data"
            result["error"] = f"Insufficient data: {result['rows']} rows"
            return result
        logger.info(f"[{pair}] Data: {len(df)} rows")
    except Exception as e:
        result["stage"] = "data"
        result["error"] = str(e)
        return result

    # 2. Compute features
    try:
        df["pair"] = pair
        df = compute_features(df, btc_df)
        latest = df.iloc[-1]
        result["atr_14"] = float(latest.get("atr_14", 0))
        result["rsi_14"] = float(latest.get("rsi_14", 50))
        result["volatility_regime"] = str(latest.get("volatility_regime", "unknown"))
        result["market_regime"] = str(latest.get("market_regime", "unknown"))
        result["volume_ratio"] = float(latest.get("volume_ratio", 0))
        logger.info(
            f"[{pair}] Features: ATR={result['atr_14']:.6f} RSI={result['rsi_14']:.1f} "
            f"Vol={result['volatility_regime']} Regime={result['market_regime']}"
        )
    except Exception as e:
        result["stage"] = "features"
        result["error"] = str(e)
        return result

    # 3. Prediction
    try:
        prediction = predictor.predict(df, pair)
        result["confidence"] = prediction.get("confidence", 0)
        result["expected_move"] = prediction.get("expected_move", 0)
        result["prob_up"] = prediction.get("prob_up", 0)
        result["prob_down"] = prediction.get("prob_down", 0)
        logger.info(
            f"[{pair}] Prediction: confidence={result['confidence']:.3f} "
            f"expected_move={result['expected_move']:.6f} "
            f"prob_up={result['prob_up']:.3f} prob_down={result['prob_down']:.3f}"
        )
    except Exception as e:
        result["stage"] = "prediction"
        result["error"] = str(e)
        return result

    # 4. Threshold check
    threshold = settings.trading.confidence_threshold
    if result["confidence"] < threshold:
        result["stage"] = "threshold"
        result["error"] = f"Confidence {result['confidence']:.3f} < threshold {threshold}"
        logger.warning(f"[{pair}] BLOCKED: {result['error']}")
        return result

    if result["prob_up"] <= result["prob_down"]:
        result["stage"] = "direction"
        result["error"] = f"Bearish: prob_up={result['prob_up']:.3f} <= prob_down={result['prob_down']:.3f}"
        logger.warning(f"[{pair}] BLOCKED: {result['error']}")
        return result

    # 5. Spread check
    try:
        _, spread_pct = fetcher.get_spread(pair)
        result["spread_pct"] = spread_pct
        logger.info(f"[{pair}] Spread: {spread_pct:.4f}")
    except Exception as e:
        result["spread_pct"] = None
        logger.warning(f"[{pair}] Could not check spread: {e}")

    result["passed"] = True
    result["stage"] = "SIGNAL_READY"
    logger.info(f"[{pair}] PASSED all checks — would generate signal")
    return result


def main():
    parser = argparse.ArgumentParser(description="Test trading flow diagnostics")
    parser.add_argument("--pair", type=str, default="BTC-USDT", help="Pair to test")
    parser.add_argument("--all-pairs", action="store_true", help="Test all top pairs")
    parser.add_argument("--model", type=str, help="Model version to load")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TFT Trading Flow Diagnostic Test")
    logger.info("=" * 60)
    logger.info(f"Confidence threshold: {settings.trading.confidence_threshold}")
    logger.info(f"Trading mode: {settings.trading.trading_mode}")

    # Initialize
    fetcher = KuCoinDataFetcher()
    predictor = TFTPredictor()

    # Load model
    if args.model:
        predictor.load(args.model)
    else:
        models = TFTPredictor.list_models()
        if not models:
            logger.error("No trained model found! Train one first with: python scripts/train_model.py")
            sys.exit(1)
        predictor.load(models[-1])
        logger.info(f"Loaded model: {models[-1]}")

    # Pre-fetch BTC data
    logger.info("Fetching BTC reference data...")
    btc_df = fetcher.fetch_klines(
        "BTC-USDT", "15min",
        start_dt=datetime.utcnow() - timedelta(hours=80),
    )
    logger.info(f"BTC data: {len(btc_df)} rows")

    # Test pairs
    if args.all_pairs:
        pairs_info = fetcher.get_top_usdt_pairs(30)
        pairs = [p["symbol"] for p in pairs_info]
        logger.info(f"Testing {len(pairs)} pairs...")
    else:
        pairs = [args.pair]

    results = []
    for pair in pairs:
        logger.info("-" * 40)
        r = test_single_pair(fetcher, predictor, pair, btc_df)
        results.append(r)

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]

    logger.info(f"Passed: {len(passed)}/{len(results)}")
    if passed:
        for r in passed:
            logger.info(f"  OK: {r['pair']} confidence={r.get('confidence', 0):.3f}")

    if failed:
        # Group failures by stage
        stage_counts = {}
        for r in failed:
            stage = r.get("stage", "unknown")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        logger.info(f"Failed by stage: {stage_counts}")
        for r in failed[:10]:  # Show first 10
            logger.info(f"  FAIL: {r['pair']} stage={r['stage']} error={r.get('error', '?')}")

    if not passed:
        logger.warning(
            "No pairs passed! Check:\n"
            "  1. Is the model trained? (python scripts/train_model.py)\n"
            "  2. Is confidence threshold too high? (current: "
            f"{settings.trading.confidence_threshold})\n"
            "  3. Are predictions returning non-zero confidence?\n"
            "  4. Run with LOG_LEVEL=DEBUG for more detail"
        )


if __name__ == "__main__":
    main()
