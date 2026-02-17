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
            self.predictor.load(model_version)
            model_loaded = True
        else:
            models = TFTPredictor.list_models()
            if models:
                self.predictor.load(models[-1])
                logger.info(f"Loaded latest model: {models[-1]}")
                model_loaded = True
            else:
                logger.warning("No trained model found. Engine will run in standby mode.")

        if model_loaded:
            # Load safety state
            self.safety.load_state()

            # Position reconciliation
            if settings.safety.position_reconciliation_on_start:
                self.executor.reconcile_positions()

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
                        models = TFTPredictor.list_models()
                        if models:
                            self.predictor.load(models[-1])
                            logger.info(f"Model found and loaded: {models[-1]}")
                            model_loaded = True
                            self.safety.load_state()
                        else:
                            logger.debug("Still waiting for a trained model...")
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
