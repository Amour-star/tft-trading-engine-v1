"""
Train PPO position management agent.
"""
import sys

sys.path.insert(0, ".")

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from data.database import init_db, register_model_version
from data.features import compute_features
from utils.logging import setup_logging


def _import_ppo():
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise ImportError("stable-baselines3 is required. Install with `pip install stable-baselines3`.") from exc
    return PPO


def _import_env():
    from tft_engine.ai.rl.environment import EnvConfig, TradeManagementEnv

    return EnvConfig, TradeManagementEnv


def load_training_data(path: Path, pair: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    if "pair" not in df.columns:
        df["pair"] = pair
    required = {"rsi", "ema_20", "volatility"}
    if required.difference(set(df.columns)):
        df = compute_features(df)
        df["rsi"] = df.get("rsi_14", 50.0)
        df["ema_20"] = df.get("ema_21", df["close"])
        df["volatility"] = df.get("volatility_20", 0.0)
    return df


def main():
    setup_logging()
    init_db()

    parser = argparse.ArgumentParser(description="Train PPO position manager")
    parser.add_argument("--pair", default="XRP-USDT", help="Pair to use")
    parser.add_argument("--timeframe", default="15min", help="Timeframe")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--output", default="models/rl/latest_ppo.zip", help="Output model path")
    args = parser.parse_args()

    data_path = Path("data/historical") / f"{args.pair}_{args.timeframe}.parquet"
    df = load_training_data(data_path, args.pair)

    EnvConfig, TradeManagementEnv = _import_env()
    env = TradeManagementEnv(
        market_data=df,
        config=EnvConfig(initial_capital=10_000.0, max_steps=min(1024, len(df) - 1)),
    )

    PPO = _import_ppo()
    model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=128, learning_rate=3e-4)
    model.learn(total_timesteps=args.timesteps)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    version = f"ppo_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    register_model_version(
        model_type="ppo",
        version=version,
        path=str(output_path),
        model_metadata={
            "pair": args.pair,
            "timeframe": args.timeframe,
            "timesteps": args.timesteps,
            "rows": int(len(df)),
        },
        activate=True,
    )

    logger.info(f"PPO model trained: version={version} rows={len(df)} timesteps={args.timesteps}")
    logger.info(f"Saved artifact: {output_path}")


if __name__ == "__main__":
    main()

