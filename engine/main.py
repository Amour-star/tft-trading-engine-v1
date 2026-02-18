"""
Main trading engine loop.
Orchestrates data fetching, AI decisions, execution, monitoring, and feedback.
"""
from __future__ import annotations

import signal
import sys
import time
from datetime import datetime
from typing import Optional

from loguru import logger

from config.settings import settings
from data.database import init_db, get_session, EngineState
from data.fetcher import KuCoinDataFetcher
from engine.decision import DecisionEngine
from engine.safety import SafetyManager
from engine.feedback import FeedbackLoop
from execution.base_executor import BaseExecutor
from execution.executor import create_executor
from execution.monitor import PositionMonitor
from models.tft_model import TFTPredictor
from utils.logging import setup_logging


class TradingEngine:
    """
    Main trading engine. Runs the signal → execute → monitor → feedback loop.
    """

    def __init__(self) -> None:
        setup_logging()
        logger.info("Initializing Trading Engine...")

        # Initialize database
        init_db()

        # Core components
        self.fetcher = KuCoinDataFetcher()
        self.predictor = TFTPredictor()
        self.decision = DecisionEngine(self.fetcher, self.predictor)
        self.safety = SafetyManager()
        self.executor: BaseExecutor = create_executor(self.fetcher)
        self.monitor = PositionMonitor(self.fetcher, self.executor, self.predictor)
        self.feedback = FeedbackLoop()

        logger.info(f"[ENGINE] Running in {settings.trading.trading_mode.upper()} mode")

        # State
        self._running: bool = False
        self._cycle_interval: int = 60  # seconds between signal scans
        self._monitor_interval: int = 10  # seconds between position checks
        self._api_errors: int = 0
        self._auto_train_attempted: bool = False

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def start(self, model_version: Optional[str] = None) -> None:
        """Start the trading engine."""
        logger.info("=" * 60)
        logger.info("TFT AI TRADING ENGINE - STARTING")
        logger.info("=" * 60)

        # Load model
        model_loaded = False
        if model_version:
            try:
                self.predictor.load(model_version)
                model_loaded = True
            except Exception as e:
                logger.error(f"Failed to load model {model_version}: {e}")
                logger.warning("Engine will continue in standby mode.")
        else:
            model_loaded = self._ensure_model_available()
            if not model_loaded:
                logger.warning("No loadable model found. Engine will run in standby mode.")

        if model_loaded:
            self._activate_after_model_load()

        # Main loop
        self._running = True
        self._save_engine_state("running", True)
        last_signal_time = 0
        last_feedback_check = 0
        last_model_check = 0

        while self._running:
            try:
                now = time.time()

                # If no model loaded, periodically check for one
                if not model_loaded:
                    if now - last_model_check >= 30:
                        last_model_check = now
                        model_loaded = self._ensure_model_available()
                        if model_loaded:
                            self._activate_after_model_load()
                        else:
                            logger.debug("Still waiting for a loadable model...")
                    time.sleep(self._monitor_interval)
                    continue

                # Monitor open position (frequent)
                closed_trade_id = self.monitor.monitor_cycle()
                if closed_trade_id:
                    # Post-trade analysis
                    self.feedback.analyze_trade(closed_trade_id)
                    session = get_session()
                    try:
                        from data.database import Trade
                        trade = session.query(Trade).filter(Trade.trade_id == closed_trade_id).first()
                        if trade and trade.pnl is not None:
                            self.safety.record_trade_result(trade.pnl, closed_trade_id)
                    finally:
                        session.close()

                # Signal generation (less frequent)
                if now - last_signal_time >= self._cycle_interval:
                    last_signal_time = now
                    self._signal_cycle()

                # Periodic feedback check
                if now - last_feedback_check >= 3600:  # hourly
                    last_feedback_check = now
                    self._feedback_cycle()

                # Check retraining
                if self.feedback.should_retrain():
                    logger.info("Model retraining recommended")
                    # In production, this would trigger async retraining
                    # For now, just log it

                time.sleep(self._monitor_interval)
                self._api_errors = 0  # Reset on success

            except KeyboardInterrupt:
                break
            except Exception as e:
                self._api_errors += 1
                logger.error(f"Engine cycle error: {e}")
                if not self.safety.check_api_stability(self._api_errors):
                    logger.critical("API instability detected - pausing")
                    self.safety.pause_trading()
                time.sleep(30)

        self._shutdown()

    def _activate_after_model_load(self) -> None:
        """Run startup hooks after a model is successfully loaded."""
        self.safety.load_state()
        if settings.safety.position_reconciliation_on_start:
            self.executor.reconcile_positions()

    def _ensure_model_available(self) -> bool:
        """
        Ensure there is a loadable model.
        Tries existing checkpoints first, then optional auto-train bootstrap.
        """
        if self._load_latest_model():
            return True

        if (
            settings.model.auto_train_if_missing
            and not self._auto_train_attempted
        ):
            self._auto_train_attempted = True
            trained_version = self._bootstrap_train_model()
            if trained_version:
                try:
                    self.predictor.load(trained_version)
                    logger.info(f"Loaded newly trained model: {trained_version}")
                    return True
                except Exception as e:
                    logger.error(f"New model load failed ({trained_version}): {e}")

        return False

    def _load_latest_model(self) -> bool:
        """Try to load available models from newest to oldest."""
        models = TFTPredictor.list_models()
        if not models:
            return False

        for version in reversed(models):
            try:
                self.predictor.load(version)
                logger.info(f"Loaded model: {version}")
                return True
            except Exception as e:
                logger.error(f"Model load failed for {version}: {e}")

        return False

    def _bootstrap_train_model(self) -> Optional[str]:
        """Train a starter model from local history, optionally fetching history first."""
        from pathlib import Path
        import pandas as pd

        from data.database import ModelMetric
        from data.features import compute_features
        from models.tft_model import train_tft

        tf = settings.model.bootstrap_timeframe
        months = settings.model.bootstrap_history_months
        pairs = settings.model.bootstrap_pairs
        data_dir = Path("data/historical")
        data_dir.mkdir(parents=True, exist_ok=True)

        data_files = sorted(data_dir.glob(f"*_{tf}.parquet"))
        if not data_files and settings.model.auto_fetch_history_if_missing:
            logger.info(
                f"No {tf} training data found. Fetching history for bootstrap "
                f"(pairs={pairs}, months={months})..."
            )
            self._fetch_bootstrap_history(data_dir, tf, months, pairs, compute_features)
            data_files = sorted(data_dir.glob(f"*_{tf}.parquet"))

        if not data_files:
            logger.warning(
                f"Auto-train skipped: no training files found for timeframe {tf} "
                f"in {data_dir}."
            )
            return None

        all_dfs = []
        for f in data_files:
            try:
                df = pd.read_parquet(f)
                if "pair" not in df.columns:
                    df["pair"] = f.stem.replace(f"_{tf}", "")
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed reading training file {f}: {e}")

        if not all_dfs:
            logger.warning("Auto-train skipped: no readable training data files.")
            return None

        combined = pd.concat(all_dfs, ignore_index=True)
        version_name = f"auto_tft_v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        logger.info(
            f"Auto-training model {version_name} on {len(combined)} rows "
            f"from {len(all_dfs)} files..."
        )
        version, metrics = train_tft(combined, model_name=version_name)

        session = get_session()
        try:
            session.query(ModelMetric).update({ModelMetric.is_active: False})
            session.add(
                ModelMetric(
                    model_version=version,
                    trained_at=datetime.utcnow(),
                    training_loss=metrics.get("training_loss"),
                    validation_loss=metrics.get("validation_loss"),
                    is_active=True,
                    notes="Auto-trained at engine startup",
                    hyperparameters={
                        "encoder_length": settings.model.encoder_length,
                        "prediction_length": settings.model.prediction_length,
                        "hidden_size": settings.model.hidden_size,
                        "attention_head_size": settings.model.attention_head_size,
                        "dropout": settings.model.dropout,
                        "learning_rate": settings.model.learning_rate,
                    },
                )
            )
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save auto-training model metrics: {e}")
        finally:
            session.close()

        logger.info(
            f"Auto-training complete: {version} (val_loss={metrics.get('validation_loss')})"
        )
        return version

    def _fetch_bootstrap_history(
        self,
        data_dir,
        timeframe: str,
        months: int,
        pairs: int,
        compute_features,
    ) -> None:
        """Fetch and save starter historical data for training."""
        import pandas as pd

        top_pairs = self.fetcher.get_top_usdt_pairs(pairs)
        pair_symbols = [p["symbol"] for p in top_pairs]
        if "BTC-USDT" not in pair_symbols:
            pair_symbols.append("BTC-USDT")

        btc_df = self.fetcher.fetch_history("BTC-USDT", timeframe, months)
        if btc_df.empty:
            logger.warning("Bootstrap fetch failed: BTC-USDT history is empty.")
            return

        for symbol in pair_symbols:
            try:
                df = self.fetcher.fetch_history(symbol, timeframe, months)
                if df.empty:
                    logger.warning(f"Bootstrap fetch: no data for {symbol} {timeframe}")
                    continue

                df["pair"] = symbol
                feat_df = compute_features(df, btc_df if not btc_df.empty else pd.DataFrame())
                out = data_dir / f"{symbol}_{timeframe}.parquet"
                feat_df.to_parquet(out, index=False)
                logger.info(f"Saved bootstrap data: {out}")
            except Exception as e:
                logger.error(f"Bootstrap fetch failed for {symbol}: {e}")

    def _signal_cycle(self) -> None:
        """Run one signal generation and execution cycle."""
        # Safety check
        can_trade, reason = self.safety.can_trade()
        if not can_trade:
            logger.info(f"Cannot trade: {reason}")
            return

        if not settings.trading.trading_enabled:
            return

        # Generate signal
        signal = self.decision.generate_signal()
        if signal is None:
            logger.debug("No valid signal generated")
            return

        logger.info(f"Signal generated: {signal.pair} | Confidence: {signal.confidence:.3f}")

        # Volatility circuit breaker
        if self.safety.check_volatility_circuit_breaker(
            signal.atr / signal.entry_price,
            0.02,  # 2% baseline volatility
        ):
            logger.warning("Circuit breaker: skipping trade")
            return

        # Execute
        balance = self.executor.get_balance()
        trade_id = self.executor.execute_signal(signal, balance)

        if trade_id:
            logger.info(f"Trade executed: {trade_id}")
        else:
            logger.warning("Trade execution failed")

    def _feedback_cycle(self) -> None:
        """Run feedback analysis and apply adjustments."""
        adjustments = self.feedback.compute_batch_adjustments()
        if adjustments:
            self.decision.update_thresholds(adjustments)
            logger.info(f"Applied threshold adjustments: {adjustments}")

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle graceful shutdown."""
        logger.info(f"Shutdown signal received ({signum})")
        self._running = False

    def _shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down Trading Engine...")
        self._save_engine_state("running", False)
        self._save_engine_state("thresholds", self.decision.get_current_thresholds())
        logger.info("Engine stopped.")

    def _save_engine_state(self, key: str, value) -> None:
        session = get_session()
        try:
            state = session.query(EngineState).filter(EngineState.key == key).first()
            if state:
                state.value = {"value": value}
                state.updated_at = datetime.utcnow()
            else:
                session.add(EngineState(key=key, value={"value": value}))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving engine state: {e}")
        finally:
            session.close()


def main() -> None:
    """Entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="TFT AI Trading Engine")
    parser.add_argument("--model", type=str, help="Model version to load")
    parser.add_argument("--dry-run", action="store_true", help="Paper trading mode (no real orders)")
    args = parser.parse_args()

    if args.dry_run and settings.trading.trading_mode.upper() != "PAPER":
        # Settings are already frozen, so we rebuild with the env var set
        import os
        os.environ["TRADING_MODE"] = "PAPER"
        import config.settings as cfg_mod
        cfg_mod.settings = cfg_mod.Settings()
        logger.info("Paper trading mode enabled via --dry-run flag")

    engine = TradingEngine()
    engine.start(model_version=args.model)


if __name__ == "__main__":
    main()
