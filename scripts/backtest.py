"""
Simple backtest runner: last 90 days performance summary.
Outputs: Sharpe, Max DD, Win Rate, Expectancy, Total Return.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import ACTIVE_SYMBOL, settings
from data.features import compute_features
from models.meta_model import XGBoostMetaModel
from models.tft_model import TFTPredictor
from utils.logging import setup_logging

DATA_DIR = ROOT / "data" / "historical"


def run_backtest(
    pair: str = ACTIVE_SYMBOL,
    timeframe: str = "15min",
    days: int = 90,
    initial_balance: float = 10_000.0,
    confidence_threshold: float = 0.45,
) -> dict:
    """Run a simple backtest on the last N days of data."""
    setup_logging()

    logger.info("=" * 60)
    logger.info(f"BACKTEST | {pair} | last {days} days | balance=${initial_balance}")
    logger.info("=" * 60)

    data_file = DATA_DIR / f"{pair}_{timeframe}.parquet"
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Run scripts/fetch_history.py first")
        return {"error": "no_data"}

    df = pd.read_parquet(data_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Filter to last N days
    cutoff = datetime.utcnow() - timedelta(days=days)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    if len(df) < 200:
        logger.error(f"Insufficient data: {len(df)} rows (need >= 200)")
        return {"error": "insufficient_data"}

    logger.info(f"Loaded {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Load models
    predictor = TFTPredictor()
    models = TFTPredictor.list_models()
    if not models:
        logger.error("No TFT model found")
        return {"error": "no_model"}

    for version in reversed(models):
        try:
            predictor.load(version)
            logger.info(f"Loaded model: {version}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {version}: {e}")
    else:
        return {"error": "model_load_failed"}

    meta_model = XGBoostMetaModel()
    meta_model.load()

    # Prepare features
    if "pair" not in df.columns:
        df["pair"] = pair
    df = compute_features(df, pd.DataFrame())
    if "pair" not in df.columns:
        df["pair"] = pair

    # Simulate trades
    lookback = settings.model.encoder_length + 10
    trades = []
    balance = initial_balance
    in_trade = False
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    entry_qty = 0.0

    for i in range(lookback, len(df) - 1, 4):
        current_close = float(df["close"].iloc[i])

        # Check exit conditions for open trade
        if in_trade:
            if current_close <= stop_price:
                pnl = (stop_price - entry_price) * entry_qty
                balance += pnl
                trades.append({"pnl": pnl, "pnl_pct": pnl / (entry_price * entry_qty)})
                in_trade = False
                continue
            elif current_close >= target_price:
                pnl = (target_price - entry_price) * entry_qty
                balance += pnl
                trades.append({"pnl": pnl, "pnl_pct": pnl / (entry_price * entry_qty)})
                in_trade = False
                continue
            else:
                continue  # Still in trade

        # Generate signal
        window = df.iloc[:i + 1].copy()
        prediction = predictor.predict(window, pair)

        if not prediction.get("valid", False):
            continue

        prob_up = float(prediction.get("prob_up", 0.0))
        prob_down = float(prediction.get("prob_down", 0.0))
        confidence = float(prediction.get("confidence", 0.0))

        if prob_up <= prob_down:
            continue

        # Meta model evaluation
        latest = window.iloc[-1]
        ts = latest.get("timestamp", datetime.utcnow())
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()

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

        final_confidence = 0.5 * confidence + 0.3 * float(meta_pred.probability) + 0.2 * 0.5

        if final_confidence < confidence_threshold:
            continue

        # Enter trade
        entry_price = current_close
        atr = float(latest.get("atr_14", entry_price * 0.01))
        if atr <= 0:
            continue

        stop_price = entry_price - atr * 2.0
        expected_move = abs(float(prediction.get("expected_move", 0.0))) * entry_price
        target_price = entry_price + max(expected_move, atr * 2.0)
        entry_qty = (balance * 0.95) / entry_price  # 95% allocation
        in_trade = True

    # Close any remaining position at last close
    if in_trade:
        final_close = float(df["close"].iloc[-1])
        pnl = (final_close - entry_price) * entry_qty
        balance += pnl
        trades.append({"pnl": pnl, "pnl_pct": pnl / (entry_price * entry_qty)})

    # Calculate metrics
    if not trades:
        logger.warning("No trades generated in backtest")
        return {
            "total_trades": 0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "total_return": 0.0,
            "final_balance": initial_balance,
        }

    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Sharpe
    if len(pnl_pcts) >= 2:
        mean_ret = float(np.mean(pnl_pcts))
        std_ret = float(np.std(pnl_pcts, ddof=1))
        sharpe = (mean_ret / std_ret * math.sqrt(96)) if std_ret > 1e-12 else 0.0
    else:
        sharpe = 0.0

    # Max Drawdown
    equity = initial_balance
    peak = equity
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    total_return = (balance - initial_balance) / initial_balance

    results = {
        "total_trades": len(trades),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "expectancy": round(expectancy, 4),
        "total_return": round(total_return, 4),
        "final_balance": round(balance, 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
    }

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Trades:       {results['total_trades']}")
    logger.info(f"  Win Rate:     {results['win_rate']:.2%}")
    logger.info(f"  Sharpe:       {results['sharpe']:.2f}")
    logger.info(f"  Max DD:       {results['max_dd']:.2%}")
    logger.info(f"  Expectancy:   ${results['expectancy']:.4f}")
    logger.info(f"  Total Return: {results['total_return']:.2%}")
    logger.info(f"  Final Balance: ${results['final_balance']:.2f}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple 90-day backtest")
    parser.add_argument("--pair", default=ACTIVE_SYMBOL, help="Trading pair")
    parser.add_argument("--days", type=int, default=90, help="Lookback days")
    parser.add_argument("--balance", type=float, default=10000, help="Starting balance")
    parser.add_argument("--threshold", type=float, default=0.45, help="Confidence threshold")
    args = parser.parse_args()

    run_backtest(
        pair=args.pair,
        days=args.days,
        initial_balance=args.balance,
        confidence_threshold=args.threshold,
    )
