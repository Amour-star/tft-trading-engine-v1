"""
End-to-end Windows-compatible smoke check for the TFT trading engine.

Runs:
1) DB init
2) Small historical data fetch
3) Quick TFT train (1 epoch)
4) One backtest pass
5) One signal generation cycle
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRADING_MODE", "PAPER")
os.environ.setdefault("KUCOIN_OFFLINE_MODE", "true")
os.environ.setdefault("FORCE_CPU", "true")
os.environ.setdefault("DATALOADER_NUM_WORKERS", "0")

from backtesting.backtester import Backtester
from config.settings import settings
from data.database import init_db
from data.fetcher import KuCoinDataFetcher
from data.features import compute_features
from engine.decision import DecisionEngine
from models.tft_model import TFTPredictor, train_tft
from utils.logging import setup_logging


def _prepare_environment() -> None:
    pass


def _fetch_small_history(data_dir: Path) -> list[Path]:
    fetcher = KuCoinDataFetcher()
    timeframe = "15min"
    months = 1

    top_pairs = fetcher.get_top_usdt_pairs(3)
    symbols = [item["symbol"] for item in top_pairs][:3]
    if "XRP-USDT" not in symbols:
        symbols.insert(0, "XRP-USDT")
    symbols = symbols[:3]

    btc_df = fetcher.fetch_history("XRP-USDT", timeframe, months)
    if btc_df.empty:
        raise RuntimeError("Failed to fetch XRP-USDT history for smoke check.")

    saved: list[Path] = []
    for symbol in symbols:
        df = fetcher.fetch_history(symbol, timeframe, months)
        if df.empty:
            continue
        df["pair"] = symbol
        features_df = compute_features(df, btc_df=btc_df if symbol != "XRP-USDT" else btc_df)
        features_df = features_df.tail(900).copy()
        if len(features_df) < (settings.model.encoder_length + settings.model.prediction_length + 20):
            continue
        out_path = data_dir / f"{symbol}_{timeframe}.parquet"
        features_df.to_parquet(out_path, index=False)
        saved.append(out_path)

    if not saved:
        raise RuntimeError("Historical data stage produced no usable parquet files.")
    return saved


def _train_quick_model(files: list[Path]) -> str:
    frames = [pd.read_parquet(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["pair", "timestamp"], inplace=True, ignore_index=True)
    version = f"smoke_tft_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    model_version, _ = train_tft(
        combined,
        model_name=version,
        max_epochs=10,
        batch_size=min(32, settings.model.batch_size),
    )
    return model_version


def _run_backtest_once(model_version: str, data_file: Path) -> None:
    df = pd.read_parquet(data_file)
    pair = str(df["pair"].iloc[0]) if "pair" in df.columns and not df.empty else "XRP-USDT"

    predictor = TFTPredictor(model_version)
    df = df.sort_values("timestamp").reset_index(drop=True)
    test_df = df.tail(240).copy()
    window = df.tail(settings.model.encoder_length + 40).copy()
    if "pair" not in window.columns:
        window["pair"] = pair
    pred = predictor.predict(window, pair)
    latest = window.iloc[-1]
    entry = float(latest["close"])
    atr = float(latest.get("atr_14", max(entry * 0.01, 1e-6)))
    stop = entry - atr * 2.0
    target = entry + max(abs(float(pred.get("expected_move", 0.0))) * entry, atr * 2.0)
    signals = [
        {
            "timestamp": pd.to_datetime(latest["timestamp"]).to_pydatetime(),
            "pair": pair,
            "side": "BUY",
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "confidence": float(pred.get("confidence", 0.5)),
            "ai_score": float(pred.get("confidence", 0.5)),
            "ai_confidence": float(pred.get("confidence", 0.5)),
        }
    ]

    bt = Backtester(initial_balance=10_000)
    _ = bt.run(test_df, signals, risk_per_trade=float(settings.trading.risk_per_trade), rl_manager=None)


def _run_signal_cycle(model_version: str) -> None:
    fetcher = KuCoinDataFetcher()
    predictor = TFTPredictor(model_version)
    decision = DecisionEngine(fetcher, predictor)
    original_get_pairs = decision.fetcher.get_top_usdt_pairs
    decision.fetcher.get_top_usdt_pairs = lambda _: original_get_pairs(1)
    _ = decision.generate_signal()


def main() -> None:
    _prepare_environment()
    setup_logging()

    logger.info("[CHECK] Initializing DB")
    init_db()

    data_dir = ROOT / "data" / "historical"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[CHECK] Fetching small historical dataset")
    files = _fetch_small_history(data_dir)

    logger.info("[CHECK] Training quick TFT model (1 epoch)")
    model_version = _train_quick_model(files)

    logger.info("[CHECK] Running one backtest pass")
    _run_backtest_once(model_version, files[0])

    logger.info("[CHECK] Running one signal generation cycle")
    _run_signal_cycle(model_version)

    print("SYSTEM STATUS: OK")


if __name__ == "__main__":
    main()

