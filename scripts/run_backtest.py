"""
Run backtesting with walk-forward validation and Monte Carlo simulation.
Uses live TFT + XGBoost meta-model signals with PPO management simulation.
"""
from __future__ import annotations

import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.backtester import Backtester
from config.settings import settings
from data.features import compute_features
from models.meta_model import XGBoostMetaModel
from models.rl_position_manager import PPOPositionManager
from models.tft_model import TFTPredictor
from utils.logging import setup_logging

DATA_DIR = ROOT / "data" / "historical"


def load_latest_tft_model() -> TFTPredictor:
    predictor = TFTPredictor()
    available = TFTPredictor.list_models()
    if not available:
        raise RuntimeError("No TFT model found. Train one first with scripts/train_model.py")

    for version in reversed(available):
        try:
            predictor.load(version)
            logger.info(f"Loaded TFT model for backtest: {version}")
            return predictor
        except Exception as e:
            logger.warning(f"Failed to load model {version}: {e}")

    raise RuntimeError("Could not load any TFT model for backtesting")


def _ensure_features(df: pd.DataFrame, pair: str, btc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    working = df.copy()
    if "pair" not in working.columns:
        working["pair"] = pair

    required = {"atr_14", "rsi_14", "volatility_20", "btc_correlation"}
    if required.difference(set(working.columns)):
        working = compute_features(working, btc_df=btc_df)
        if "pair" not in working.columns:
            working["pair"] = pair
    return working


def make_ai_signal_generator(
    pair: str,
    predictor: TFTPredictor,
    meta_model: XGBoostMetaModel,
    min_confidence: float,
    max_evaluations: int | None = None,
):
    lookback = settings.model.encoder_length + 5

    def generate_ai_signals(train_df: pd.DataFrame, test_df: pd.DataFrame):
        signals = []
        eval_count = 0
        if test_df.empty or len(test_df) < lookback + 1:
            return signals

        train_working = train_df.copy()
        test_working = test_df.copy()
        if "pair" not in train_working.columns:
            train_working["pair"] = pair
        if "pair" not in test_working.columns:
            test_working["pair"] = pair

        btc_df = pd.DataFrame()
        if pair != "XRP-USDT":
            btc_file = DATA_DIR / f"XRP-USDT_15min.parquet"
            if btc_file.exists():
                btc_df = pd.read_parquet(btc_file)

        history = _ensure_features(pd.concat([train_working, test_working], ignore_index=True), pair, btc_df=btc_df)
        train_len = len(train_working)

        # Evaluate every 4 candles to limit overtrading.
        for idx in range(train_len + lookback, len(history), 4):
            if max_evaluations is not None and eval_count >= max_evaluations:
                break
            eval_count += 1
            window = history.iloc[: idx + 1].copy()
            window["pair"] = pair

            prediction = predictor.predict(window, pair)
            if prediction.get("prob_up", 0.0) <= prediction.get("prob_down", 0.0):
                continue

            latest = window.iloc[-1]
            ts = latest.get("timestamp", datetime.utcnow())
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            elif not isinstance(ts, datetime):
                ts = datetime.utcnow()

            snapshot = {
                "atr_14": float(latest.get("atr_14", 0.0)),
                "rsi_14": float(latest.get("rsi_14", 50.0)),
                "volatility_20": float(latest.get("volatility_20", 0.0)),
                "volatility_regime": str(latest.get("volatility_regime", "normal")),
                "btc_correlation": float(latest.get("btc_correlation", 0.0)),
                "hour_sin": float(latest.get("hour_sin", 0.0)),
                "hour_cos": float(latest.get("hour_cos", 0.0)),
                "dow_sin": float(latest.get("dow_sin", 0.0)),
                "dow_cos": float(latest.get("dow_cos", 0.0)),
            }
            meta_features = meta_model.build_features(prediction, snapshot, ts)
            meta_pred = meta_model.predict(prediction, meta_features)

            if meta_pred.probability < min_confidence:
                continue

            entry = float(latest["close"])
            atr = float(latest.get("atr_14", entry * 0.01))
            if atr <= 0:
                continue

            stop = entry - atr * 2.0
            expected_move_abs = abs(float(prediction.get("expected_move", 0.0))) * entry
            target = entry + max(expected_move_abs, atr * 2.0)

            signals.append(
                {
                    "timestamp": ts,
                    "pair": pair,
                    "side": "BUY",
                    "entry_price": entry,
                    "stop_price": stop,
                    "target_price": target,
                    "confidence": float(meta_pred.probability),
                    "ai_score": float(meta_pred.probability),
                    "ai_confidence": float(meta_pred.confidence_score),
                }
            )

        return signals

    return generate_ai_signals


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Run AI-aligned backtest")
    parser.add_argument("--pair", default="XRP-USDT", help="Pair to backtest")
    parser.add_argument("--timeframe", default="15min", help="Timeframe")
    parser.add_argument("--balance", type=float, default=10000, help="Starting balance")
    parser.add_argument("--monte-carlo", type=int, default=1000, help="Monte Carlo simulations")
    parser.add_argument("--min-confidence", type=float, default=None, help="Meta-model confidence threshold")
    parser.add_argument("--window-size", type=int, default=2000, help="Walk-forward train window")
    parser.add_argument("--step-size", type=int, default=500, help="Walk-forward step size")
    parser.add_argument("--max-evaluations", type=int, default=12, help="Max signal evaluations per walk-forward step")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("AI-ALIGNED BACKTEST ENGINE")
    logger.info("=" * 60)

    data_file = DATA_DIR / f"{args.pair}_{args.timeframe}.parquet"
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Run scripts/fetch_history.py first")
        return

    df = pd.read_parquet(data_file)
    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise RuntimeError(f"Backtest data missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    if "pair" not in df.columns:
        df["pair"] = args.pair
    min_rows = settings.model.encoder_length + settings.model.prediction_length + 100
    if len(df) < min_rows:
        raise RuntimeError(f"Backtest requires at least {min_rows} rows, found {len(df)}.")
    logger.info(f"Loaded {len(df)} candles for {args.pair} {args.timeframe}")

    predictor = load_latest_tft_model()
    meta_model = XGBoostMetaModel()
    meta_loaded = meta_model.load()
    if meta_loaded:
        logger.info(f"Loaded XGBoost meta-model: {meta_model.model_version}")
    else:
        logger.warning("XGBoost meta-model not found, falling back to TFT confidence.")

    rl_manager = PPOPositionManager()
    rl_loaded = rl_manager.load()
    if rl_loaded:
        logger.info(f"Loaded PPO position manager: {rl_manager.model_version}")
    else:
        logger.warning("PPO model not found, using heuristic RL fallback.")

    min_confidence = (
        float(args.min_confidence)
        if args.min_confidence is not None
        else float(settings.trading.confidence_threshold)
    )

    signal_generator = make_ai_signal_generator(
        pair=args.pair,
        predictor=predictor,
        meta_model=meta_model,
        min_confidence=min_confidence,
        max_evaluations=max(1, int(args.max_evaluations)),
    )

    bt = Backtester(initial_balance=args.balance)

    logger.info("\n--- Walk-Forward Validation (TFT + XGB + PPO sim) ---")
    wf_result = bt.walk_forward(
        df=df,
        signal_generator=signal_generator,
        window_size=max(args.window_size, settings.model.encoder_length + 100),
        step_size=max(50, args.step_size),
        risk_per_trade=settings.trading.risk_per_trade,
        rl_manager=rl_manager,
    )

    logger.info("\nWalk-Forward Results:")
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

    if wf_result.trades:
        logger.info(f"\n--- Monte Carlo ({args.monte_carlo} simulations) ---")
        mc_result = bt.monte_carlo(wf_result.trades, args.monte_carlo)

        logger.info("\nMonte Carlo Results:")
        logger.info(f"  Median Final Balance: ${mc_result['median_final_balance']:.2f}")
        logger.info(f"  P5 Balance: ${mc_result['p5_final_balance']:.2f}")
        logger.info(f"  P95 Balance: ${mc_result['p95_final_balance']:.2f}")
        logger.info(f"  Median Sharpe: {mc_result['median_sharpe']:.2f}")
        logger.info(f"  P5 Sharpe: {mc_result['p5_sharpe']:.2f}")
        logger.info(f"  P95 Max Drawdown: {mc_result['p95_max_drawdown']:.2%}")
        logger.info(f"  Prob Profit: {mc_result['probability_profit']:.2%}")
        logger.info(f"  Prob Ruin: {mc_result['probability_ruin']:.2%}")
        logger.info(f"  PASSED: {mc_result['passed']}")

    all_passed = wf_result.passed and (not wf_result.trades or mc_result.get("passed", False))
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("BACKTEST PASSED - System approved for live trading")
    else:
        logger.warning("BACKTEST FAILED - DO NOT proceed to live trading")
        raise SystemExit(2)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

