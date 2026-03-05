"""
Main trading engine runtime and shutdown orchestration.
"""
from __future__ import annotations

import gc
import math
import os
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from loguru import logger


@dataclass
class EngineRuntimeStats:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    _r_sum: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.win_rate = 0.0
        self.avg_r = 0.0
        self.total_pnl = 0.0
        self.unrealized_pnl = 0.0
        self._r_sum = 0.0

    def record_trade(self, pnl: float, r_multiple: float) -> None:
        pnl_value = float(pnl or 0.0)
        r_value = float(r_multiple or 0.0)
        self.total_trades += 1
        if pnl_value > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.total_pnl += pnl_value
        self._r_sum += r_value
        self.avg_r = self._r_sum / self.total_trades if self.total_trades > 0 else 0.0
        self.win_rate = self.wins / self.total_trades if self.total_trades > 0 else 0.0

    def snapshot_unrealized(self, unrealized: float) -> None:
        self.unrealized_pnl = float(unrealized or 0.0)

from config.settings import ACTIVE_SYMBOL, BASE_DIR, TRADING_UNIVERSE, settings
from core.instance_lock import InstanceLock
from core.shutdown_controller import shutdown_controller
from data.database import (
    DecisionEvent,
    EquityHistory,
    EngineState,
    MetricSnapshot,
    Statistics,
    Trade,
    dispose_engine,
    get_session,
    init_db,
    register_model_version,
    remove_session,
)
from data.fetcher import KuCoinDataFetcher
from engine.attribution import update_agent_performance
from engine.decision import DecisionEngine
from engine.feedback import FeedbackLoop
from engine.governance import GovernanceService
from engine.health import HealthServer
from engine.metrics import PerformanceTracker
from engine.performance_metrics import update_risk_metrics_snapshot
from engine.safety import SafetyManager
from engine.strategy_evolution import StrategyEvolutionEngine
from execution.base_executor import BaseExecutor
from execution.event_bus import publish_event
from execution.executor import create_executor
from execution.monitor import PositionMonitor
from models.rl_position_manager import PPOPositionManager
from models.tft_model import HAS_TORCH, TFTPredictor
from risk.prop_risk_manager import PropRiskManager
from risk.safety_layer import evaluate_and_arm_kill_switch, is_safe_mode
from services.reconciliation import ReconciliationProcessor
from utils.logging import setup_logging


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_ppo_score(value: float) -> float:
    return _clamp01(float(value) / 2.0)


def _finite_or_none(value: object, digits: Optional[int] = None) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits) if digits is not None else numeric


MAX_TFT_RETRIES = 1


class TradingEngine:
    """
    Main trading engine. Runs the signal -> execute -> monitor -> feedback loop.
    """

    def __init__(self) -> None:
        setup_logging()
        logger.info(f"Engine initialized for SYMBOL={ACTIVE_SYMBOL}")
        logger.info(f"Initializing Trading Engine for {ACTIVE_SYMBOL}...")
        logger.info(f"Universe locked to {ACTIVE_SYMBOL}")

        # Initialize database.
        init_db()

        # GPU auto-detection and optimization (runtime).
        device = "cpu"
        gpu_name: Optional[str] = None
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                torch.set_float32_matmul_precision("high")
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = "Unknown GPU"
                logger.info(f"[GPU] CUDA enabled | float32_matmul_precision=high | device={gpu_name}")
            else:
                logger.info("[GPU] CUDA not available, using CPU")
        except Exception:
            device = "cpu"
            gpu_name = None

        self.device = device
        logger.bind(event="ENV_DEVICE", device=device, gpu_name=gpu_name).info(
            "[ENV] Device: {device}", device=device
        )
        if gpu_name:
            logger.bind(event="ENV_GPU", gpu_name=gpu_name).info("[ENV] GPU Name: {gpu}", gpu=gpu_name)
        self._save_engine_state("device", device)

        self.mode = settings.trading.trading_mode.upper()
        logger.info(f"[ENGINE] Running in {self.mode} mode")

        # Core components.
        self.fetcher = KuCoinDataFetcher()
        self.predictor = TFTPredictor()
        self.decision = DecisionEngine(self.fetcher, self.predictor)
        self.safety = SafetyManager()
        self.executor: BaseExecutor = create_executor(self.fetcher)
        self.rl_manager = PPOPositionManager()
        self.monitor = PositionMonitor(
            self.fetcher,
            self.executor,
            self.predictor,
            rl_manager=self.rl_manager,
        )
        self.reconciliation = ReconciliationProcessor(
            self.executor,
            poll_interval_seconds=max(
                0.2,
                float(self._env_int("RECONCILIATION_POLL_SECONDS", 2)),
            ),
        )
        self.feedback = FeedbackLoop()
        self.governance = GovernanceService()
        self.strategy_evolution = StrategyEvolutionEngine()
        self.prop_risk = PropRiskManager(self.fetcher, self.executor)
        self.instance_lock = InstanceLock(
            os.getenv("ENGINE_LOCK_FILE", str(BASE_DIR / "data" / "tft_engine.lock"))
        )
        self.strategy_evolution.apply_to_decision(self.decision)

        self.shutdown_controller = shutdown_controller
        self.shutdown_controller.bind_signals()

        # Runtime state.
        self._market_interval: int = max(2, self._env_int("ENGINE_MARKET_INTERVAL_SECONDS", 10))
        self._signal_interval: int = max(30, self._env_int("ENGINE_SIGNAL_INTERVAL_SECONDS", 60))
        self._secondary_signal_interval: int = max(
            self._signal_interval,
            self._env_int("ENGINE_SECONDARY_SIGNAL_INTERVAL_SECONDS", 300),
        )
        self._cycle_interval: int = self._signal_interval
        self._monitor_interval: int = max(1, self._env_int("ENGINE_MONITOR_INTERVAL_SECONDS", 10))
        self._snapshot_interval: int = max(5, self._env_int("ENGINE_SNAPSHOT_INTERVAL_SECONDS", 10))
        self._align_signal_to_interval: bool = self._env_bool("ENGINE_ALIGN_SIGNAL_TO_INTERVAL", True)
        self._watchdog_missed_cycles: int = max(2, self._env_int("ENGINE_WATCHDOG_MISSED_CYCLES", 2))
        self._api_errors: int = 0
        self._auto_train_attempted: bool = False
        self._accept_new_trades: bool = True
        self._market_data_status: Dict[str, Any] = self.fetcher.get_market_data_status(ACTIVE_SYMBOL)
        self._last_signal_bucket: Optional[int] = None  # backward compatibility field
        self._cycle_buckets: Dict[str, int] = {}
        self._last_signal_cycle_monotonic: float = time.monotonic()
        self._watchdog_last_warning_monotonic: float = 0.0
        self._last_snapshot_mono: float = 0.0
        self._last_cycle_reason: str = "startup"
        self._no_trade_reason_counts: Dict[str, int] = {}

        self._health_ready = False
        self._health_started_at = time.monotonic()
        self._health_server: Optional[HealthServer] = None
        self._start_health_server()
        self._save_engine_state("accept_new_trades", self._accept_new_trades)
        self._save_engine_state("market_data_status", self._market_data_status)

        startup_diag_raw = self.fetcher.startup_diagnostics(self.mode)
        startup_diag = startup_diag_raw if isinstance(startup_diag_raw, dict) else {}
        self._market_data_status = dict(startup_diag)
        self._save_engine_state("market_data_status", self._market_data_status)
        if not bool(startup_diag.get("can_trade", True)):
            self._accept_new_trades = False
            startup_error = str(
                startup_diag.get("startup_error")
                or "Market-data startup validation failed; new trades are disabled."
            )
            self._last_cycle_reason = "market_data_startup_check_failed"
            self._save_engine_state("accept_new_trades", self._accept_new_trades)
            self._save_engine_state("last_cycle_reason", self._last_cycle_reason)
            logger.error(
                "[ENGINE] Startup check blocked trading for {}: {}",
                ACTIVE_SYMBOL,
                startup_error,
            )

        self._tft_retries: int = 0
        self._tft_disabled_notice_emitted: bool = False
        self._tft_retry_backoff_seconds: int = max(
            60,
            self._env_int("TFT_MODEL_RETRY_BACKOFF_SECONDS", 900),
        )
        self._next_tft_retry_monotonic: float = 0.0
        self._tft_scan_limit_per_cycle: int = max(
            1,
            self._env_int("TFT_MODEL_SCAN_LIMIT_PER_CYCLE", 16),
        )
        self._symbol_slug: str = str(os.getenv("SYMBOL", ACTIVE_SYMBOL)).split("-")[0].lower()
        self._tft_disable_flag_path = Path(
            os.getenv("TFT_DISABLE_FLAG_PATH", "/app/state/tft_disabled.flag")
        )
        # Only respect env var, not filesystem flags
        force_disable = os.getenv("TFT_FORCE_DISABLE", "").strip().lower() in ("true", "1", "yes")
        self._tft_disabled = force_disable
        if force_disable:
            logger.warning(
                "TFT disabled by TFT_FORCE_DISABLE env var for {}", self._symbol_slug
            )
            self._apply_xgb_fallback_mode()
        elif self._tft_disable_flag_path.exists():
            logger.info(
                "Stale TFT disable flag found for {} at {}. Cleaning up (not enforced without TFT_FORCE_DISABLE).",
                self._symbol_slug,
                self._tft_disable_flag_path,
            )
            try:
                self._tft_disable_flag_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Could not remove stale TFT flag: {}", exc)

        self._engine_state: str = "INIT"  # INIT, LOADING_MODEL, ACTIVE, DEGRADED_FALLBACK, HALTED
        self._state_lock = threading.Lock()
        self._cycle_in_progress: bool = False
        self._inference_in_progress: bool = False
        self._trade_execution_in_progress: bool = False
        self.open_positions: List[str] = []
        self.trade_history: List[str] = []
        self.stats = EngineRuntimeStats()
        logger.info(
            "[ENGINE] Cadence market={}s signal={}s secondary={}s monitor={}s snapshot={}s align={} watchdog={} cycles",
            self._market_interval,
            self._signal_interval,
            self._secondary_signal_interval,
            self._monitor_interval,
            self._snapshot_interval,
            self._align_signal_to_interval,
            self._watchdog_missed_cycles,
        )

    def _start_health_server(self) -> None:
        host = os.getenv("ENGINE_HEALTH_HOST", "0.0.0.0")
        port_raw = os.getenv("ENGINE_HEALTH_PORT", "8000").strip()
        try:
            port = int(port_raw)
        except ValueError:
            port = 8000
        try:
            self._health_server = HealthServer(host, port, self._health_payload)
            self._health_server.start()
            logger.info("[HEALTH] Health server listening on {}:{}", host, port)
        except Exception as exc:
            self._health_server = None
            logger.error("[HEALTH] Failed to start health server: {}", exc)

    def _health_payload(self) -> Dict[str, object]:
        uptime_seconds = int(max(0.0, time.monotonic() - self._health_started_at))
        market_status = self.fetcher.get_market_data_status(ACTIVE_SYMBOL)
        self._market_data_status = market_status
        return {
            "status": "ok" if self._health_ready else "starting",
            "ready": self._health_ready,
            "mode": self.mode.lower(),
            "engine_state": getattr(self, "_engine_state", "UNKNOWN"),
            "model_loaded": bool(self.predictor.model is not None) if hasattr(self, "predictor") else False,
            "model_version": getattr(self.predictor, "model_version", "") if hasattr(self, "predictor") else "",
            "tft_disabled": getattr(self, "_tft_disabled", False),
            "last_cycle_reason": self._last_cycle_reason,
            "uptime_seconds": uptime_seconds,
            "symbol": ACTIVE_SYMBOL,
            "market_data_source": market_status.get("market_data_source", "public_ticker"),
            "ticker_source": market_status.get("ticker_source", "unknown"),
            "orderbook_source": market_status.get("orderbook_source", "unknown"),
            "synthetic_active": bool(market_status.get("synthetic_active", False)),
            "accept_new_trades": bool(self._accept_new_trades),
            "startup_error": market_status.get("startup_error", ""),
        }

    def start(self, model_version: Optional[str] = None) -> None:
        """Start the trading engine."""
        self.instance_lock.acquire()
        try:
            logger.info("=" * 60)
            logger.info("TFT AI TRADING ENGINE - STARTING")
            logger.info("=" * 60)

            self._engine_state = "LOADING_MODEL"
            self._health_ready = True
            self.reconciliation.start()

            model_loaded = self._load_startup_model(model_version)
            if model_loaded:
                model_loaded = self._activate_after_model_load()
                if not model_loaded:
                    logger.warning(
                        f"Loaded model is incompatible with {ACTIVE_SYMBOL} runtime; "
                        f"entering DEGRADED_FALLBACK mode. Trading continues with XGB only."
                    )
                    self._engine_state = "DEGRADED_FALLBACK"
                    self._apply_xgb_fallback_mode()
                else:
                    self._engine_state = "ACTIVE"
            else:
                self._engine_state = "DEGRADED_FALLBACK"

            logger.bind(
                event="ENGINE_STATE_RESOLVED",
                engine_state=self._engine_state,
                model_loaded=model_loaded,
                model_version=self.predictor.model_version or "none",
                symbol=ACTIVE_SYMBOL,
                mode=self.mode,
            ).info("ENGINE_STATE_RESOLVED")
            if self._engine_state == "DEGRADED_FALLBACK":
                logger.bind(event="FALLBACK_MODE_ACTIVE", symbol=ACTIVE_SYMBOL).warning(
                    "FALLBACK_MODE_ACTIVE"
                )

            self._save_engine_state("running", True)
            self._restore_runtime_state()
            last_signal_time = 0.0
            last_secondary_signal_time = 0.0
            last_market_time = 0.0
            last_feedback_check = 0.0
            last_model_check = 0.0

            while not self.shutdown_controller.should_stop():
                try:
                    with self._cycle_scope():
                        (
                            model_loaded,
                            last_signal_time,
                            last_secondary_signal_time,
                            last_market_time,
                            last_feedback_check,
                            last_model_check,
                        ) = self._run_trading_cycle(
                            model_loaded=model_loaded,
                            last_signal_time=last_signal_time,
                            last_secondary_signal_time=last_secondary_signal_time,
                            last_market_time=last_market_time,
                            last_feedback_check=last_feedback_check,
                            last_model_check=last_model_check,
                        )
                    self._api_errors = 0
                except Exception as exc:
                    self._api_errors += 1
                    logger.exception("[ENGINE] Trading cycle error: {}", exc)
                    if not self.safety.check_api_stability(self._api_errors):
                        logger.critical("[ENGINE] API instability detected - pausing trading")
                        self.safety.pause_trading()
                    if self.shutdown_controller.should_stop():
                        break
                    self._sleep_interruptible(1)

                if self.shutdown_controller.should_stop():
                    logger.info("[ENGINE] Finishing current cycle")
                    break

                self._sleep_interruptible(self._monitor_interval)
        finally:
            self._shutdown()

    def _load_startup_model(self, model_version: Optional[str]) -> bool:
        if not HAS_TORCH:
            logger.error("Torch not installed. Install compatible version.")
            logger.warning("Engine will continue in DEGRADED_FALLBACK mode (XGB only).")
            self._engine_state = "DEGRADED_FALLBACK"
            self._apply_xgb_fallback_mode()
            return False

        model_loaded = False
        if model_version:
            try:
                self.predictor.load(model_version)
                model_loaded = True
            except Exception as exc:
                logger.error(f"Failed to load model {model_version}: {exc}")
                logger.warning("Engine will continue in DEGRADED_FALLBACK mode (XGB only).")
                self._engine_state = "DEGRADED_FALLBACK"
                self._apply_xgb_fallback_mode()
            return model_loaded

        model_loaded = self._ensure_model_available()
        if not model_loaded:
            logger.warning(
                "No loadable TFT model found. Engine entering DEGRADED_FALLBACK mode. "
                "Trading continues with XGB meta-model only."
            )
            self._engine_state = "DEGRADED_FALLBACK"
            self._apply_xgb_fallback_mode()
        else:
            self._engine_state = "ACTIVE"
        return model_loaded

    def _run_trading_cycle(
        self,
        model_loaded: bool,
        last_signal_time: float,
        last_secondary_signal_time: float,
        last_market_time: float,
        last_feedback_check: float,
        last_model_check: float,
    ) -> tuple[bool, float, float, float, float, float]:
        now = time.monotonic()
        now_dt = datetime.utcnow()
        self._handle_pending_hard_reset()
        self._handle_pending_paper_reset()

        # If no model loaded, periodically check for one — but STILL TRADE using fallback.
        if not model_loaded:
            if now - last_model_check >= 60:
                last_model_check = now
                if now < self._next_tft_retry_monotonic:
                    remaining = max(0, int(self._next_tft_retry_monotonic - now))
                    logger.debug(
                        "TFT reload backoff active for {} ({}s remaining). Trading in DEGRADED_FALLBACK mode.",
                        self._symbol_slug,
                        remaining,
                    )
                else:
                    model_loaded = self._ensure_model_available()
                    if model_loaded:
                        model_loaded = self._activate_after_model_load()
                        if model_loaded:
                            self._engine_state = "ACTIVE"
                            logger.info("TFT model loaded successfully. Engine state: ACTIVE")
                        else:
                            logger.warning(
                                f"Loaded model failed {ACTIVE_SYMBOL} vocabulary startup checks; "
                                f"continuing in DEGRADED_FALLBACK mode."
                            )
                            self._engine_state = "DEGRADED_FALLBACK"
                    else:
                        logger.debug("Still waiting for a loadable TFT model. Trading in DEGRADED_FALLBACK mode.")
            # CRITICAL: Do NOT return early — fall through to signal generation
            # so the engine trades using XGB fallback even without TFT.

        run_market, updated_last_market_time = self._should_run_timed_cycle(
            key="market_data",
            interval_seconds=self._market_interval,
            now_dt=now_dt,
            now_mono=now,
            last_run_time=last_market_time,
        )
        if run_market:
            last_market_time = updated_last_market_time
            self._market_data_cycle()

        # Monitor open position (frequent).
        with self._inference_scope():
            closed_trade_id = self.monitor.monitor_cycle()
        if closed_trade_id:
            self._handle_closed_trade(closed_trade_id)

        # Shutdown is checked before opening any new trade cycle.
        if self.shutdown_controller.should_stop():
            return (
                model_loaded,
                last_signal_time,
                last_secondary_signal_time,
                last_market_time,
                last_feedback_check,
                last_model_check,
            )

        run_signal, updated_last_signal_time = self._should_run_timed_cycle(
            key="primary_signal",
            interval_seconds=self._signal_interval,
            now_dt=now_dt,
            now_mono=now,
            last_run_time=last_signal_time,
        )
        if run_signal:
            last_signal_time = updated_last_signal_time
            self._signal_cycle(cycle_label="1m")

        run_secondary, updated_last_secondary = self._should_run_timed_cycle(
            key="secondary_signal",
            interval_seconds=self._secondary_signal_interval,
            now_dt=now_dt,
            now_mono=now,
            last_run_time=last_secondary_signal_time,
        )
        if run_secondary:
            last_secondary_signal_time = updated_last_secondary
            self._signal_cycle(cycle_label="5m")

        self._watchdog_check(now)
        if now - self._last_snapshot_mono >= self._snapshot_interval:
            self._last_snapshot_mono = now
            self._persist_runtime_snapshots()

        # Periodic feedback check.
        if now - last_feedback_check >= 3600:
            last_feedback_check = now
            self._feedback_cycle()

        # Retraining check: log at most once per hour, never block trading
        if not hasattr(self, '_last_retrain_log_time'):
            self._last_retrain_log_time = 0.0
        if self.feedback.should_retrain():
            if now - self._last_retrain_log_time >= 3600:
                self._last_retrain_log_time = now
                logger.info("Model retraining recommended (logged once/hour, does not block trading)")

        return (
            model_loaded,
            last_signal_time,
            last_secondary_signal_time,
            last_market_time,
            last_feedback_check,
            last_model_check,
        )

    def _handle_closed_trade(self, closed_trade_id: str) -> None:
        self.feedback.analyze_trade(closed_trade_id)
        session = get_session()
        try:
            from data.database import Trade

            trade = session.query(Trade).filter(Trade.trade_id == closed_trade_id).first()
            if not trade or trade.pnl is None:
                return

            # Extract scalar values before any downstream call closes the
            # scoped session and detaches this ORM instance.
            trade_pnl = float(trade.pnl)
            trade_r_multiple = float(trade.r_multiple or 0.0)
        finally:
            session.close()

        # All downstream helpers open (and close) their own sessions,
        # so we must NOT access the `trade` ORM object after this point.
        self.safety.record_trade_result(trade_pnl, closed_trade_id)
        update_agent_performance(closed_trade_id)
        self.update_risk_metrics()
        self.prop_risk.on_trade_closed_by_id(closed_trade_id)

        if closed_trade_id in self.open_positions:
            self.open_positions.remove(closed_trade_id)
        self.trade_history.append(closed_trade_id)
        self.stats.record_trade(trade_pnl, trade_r_multiple)
        self._persist_runtime_snapshots()

        evolved = self.strategy_evolution.evolve_if_due(
            open_trade_count=self._count_open_trades()
        )
        if evolved:
            self.decision.update_thresholds(evolved)
            self._save_engine_state("thresholds", self.decision.get_current_thresholds())
            logger.info(f"[EVOLVE] Applied strategy evolution: {evolved}")

    def _handle_pending_paper_reset(self) -> None:
        """Reload executor state when a reset is pending."""
        if settings.trading.trading_mode.upper() != "PAPER":
            return
        session = get_session()
        try:
            state = session.query(EngineState).filter(EngineState.key == "paper_reset_pending").first()
            if not state or not state.value.get("value"):
                return
            pending_balance = state.value.get("initial_balance")
            logger.info("Reloading paper executor state after reset (initial={})", pending_balance)
            self.executor.reload_runtime_state()
            self.safety.load_state(force_refresh=True)
            self.safety.reset_consecutive_losses()

            baseline_equity = float(
                pending_balance
                if pending_balance is not None
                else self.executor.get_balance()
            )
            now_dt = datetime.utcnow()
            today = now_dt.date().isoformat()
            rebaseline_payload = {
                "value": True,
                "reason": "paper_reset_applied",
                "timestamp": now_dt.isoformat(),
            }

            def _upsert_state(key: str, payload: Dict[str, Any]) -> None:
                row = session.query(EngineState).filter(EngineState.key == key).first()
                if row:
                    row.value = payload
                    row.updated_at = now_dt
                else:
                    session.add(EngineState(key=key, value=payload))

            # Re-arm baselines using post-reset equity to avoid stale kill-switch trips.
            _upsert_state("kill_switch_day_start_equity", {"date": today, "equity": baseline_equity})
            _upsert_state("kill_switch_initial_equity", {"equity": baseline_equity})
            _upsert_state("kill_switch_rebaseline", rebaseline_payload)
            _upsert_state("prop_day_start_equity", {"value": {"date": today, "equity": baseline_equity}})
            _upsert_state("prop_max_equity", {"value": baseline_equity})
            _upsert_state("single_symbol_loss_streak", {"value": 0})
            _upsert_state("prop_cooldown_until", {"value": None})
            _upsert_state("prop_rebaseline_requested", rebaseline_payload)

            state.value = {"value": False}
            state.updated_at = now_dt
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Failed to apply pending paper reset: {}", exc)
            # Prevent reset-loop storms if rebaseline writes fail.
            try:
                state = session.query(EngineState).filter(EngineState.key == "paper_reset_pending").first()
                if state:
                    state.value = {"value": False, "error": str(exc)}
                    state.updated_at = datetime.utcnow()
                    session.commit()
            except Exception:
                session.rollback()
        finally:
            session.close()

    def _clear_runtime_state(self) -> None:
        if self.open_positions:
            self.open_positions.clear()
        if self.trade_history:
            self.trade_history.clear()
        self.stats.reset()
        logger.info("[ENGINE] Runtime caches cleared")

    def _handle_pending_hard_reset(self) -> None:
        session = get_session()
        try:
            state = session.query(EngineState).filter(EngineState.key == "hard_reset_pending").first()
            if not state or not state.value.get("value"):
                return
            logger.warning("Applying pending hard reset cleanup")
            self._clear_runtime_state()
            state.value = {"value": False}
            state.updated_at = datetime.utcnow()
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Failed to apply pending hard reset cleanup: {}", exc)
        finally:
            session.close()

    def _model_supports_symbol(self) -> bool:
        """Validate loaded model vocabulary for the active symbol."""
        try:
            supported_pairs = self.decision._get_supported_pairs()
        except Exception as exc:
            logger.error("Could not inspect model pair vocabulary: {}", exc)
            return False

        supported_list = sorted(supported_pairs)
        logger.info("Model supports: {}", supported_list)
        if ACTIVE_SYMBOL not in supported_pairs:
            logger.error(
                "Loaded model '{}' excludes required symbol {}",
                self.predictor.model_version or "unknown",
                ACTIVE_SYMBOL,
            )
            return False
        return True

    def _activate_after_model_load(self) -> bool:
        """Run startup hooks after a model is successfully loaded."""
        if not self._model_supports_symbol():
            return False
        self._tft_retries = 0
        self._next_tft_retry_monotonic = 0.0
        self.safety.load_state()
        self.strategy_evolution.apply_to_decision(self.decision)
        self._save_engine_state("thresholds", self.decision.get_current_thresholds())
        return True

    def _ensure_model_available(self) -> bool:
        """
        Ensure there is a loadable model.
        Tries existing checkpoints first, then optional auto-train bootstrap.
        """
        if self._is_tft_disabled():
            return False

        if self._load_latest_model():
            return True

        if settings.model.auto_train_if_missing and not self._auto_train_attempted:
            self._auto_train_attempted = True
            try:
                trained_version = self._bootstrap_train_model()
            except Exception as exc:
                logger.error(f"Auto-train bootstrap failed: {exc}")
                return False
            if trained_version:
                try:
                    self.predictor.load(trained_version)
                    if not self._model_supports_symbol():
                        logger.error(
                            "Auto-trained model {} does not include {} in vocabulary.",
                            trained_version,
                            ACTIVE_SYMBOL,
                        )
                        return False
                    logger.info(f"Loaded newly trained model: {trained_version}")
                    return True
                except Exception as exc:
                    logger.error(f"New model load failed ({trained_version}): {exc}")

        return False

    def _load_latest_model(self) -> bool:
        """Try to load available models from newest to oldest."""
        if self._is_tft_disabled():
            return False

        models = TFTPredictor.list_models()
        if not models:
            return False

        scanned = 0
        symbol_mismatch_count = 0
        for version in reversed(models):
            if scanned >= self._tft_scan_limit_per_cycle:
                logger.debug(
                    "TFT scan limit reached for {} (limit={} per cycle).",
                    self._symbol_slug,
                    self._tft_scan_limit_per_cycle,
                )
                break
            scanned += 1
            try:
                self.predictor.load(version)
                if not self._model_supports_symbol():
                    symbol_mismatch_count += 1
                    logger.warning(
                        "Skipping model {} because it does not include {} in vocabulary.",
                        version,
                        ACTIVE_SYMBOL,
                    )
                    try:
                        self.predictor.model = None
                        self.predictor.model_version = "tft_disabled"
                    except Exception:
                        pass
                    continue
                logger.info(f"Loaded model: {version}")
                return True
            except ImportError as exc:
                self._on_tft_load_failure(version, exc, import_related=True)
                return False
            except Exception as exc:
                message = str(exc).lower()
                import_related = "pytorch-forecasting required" in message
                self._on_tft_load_failure(version, exc, import_related=import_related)
                if import_related or self._tft_retries > MAX_TFT_RETRIES:
                    return False

        if scanned > 0 and scanned == symbol_mismatch_count:
            self._disable_tft_permanently(reason=f"no_compatible_model_for_{ACTIVE_SYMBOL}")

        try:
            self.predictor.model = None
            self.predictor.model_version = "tft_disabled"
        except Exception:
            pass
        return False

    def _on_tft_load_failure(self, version: str, exc: Exception, import_related: bool) -> None:
        self._tft_retries += 1
        if import_related:
            logger.warning(
                "TFT unavailable for model {} ({}). Switching to fallback mode.",
                version,
                exc,
            )
            self._apply_xgb_fallback_mode()
        else:
            logger.error("Model load failed for {}: {}", version, exc)

        if self._tft_retries > MAX_TFT_RETRIES:
            self._disable_tft_permanently(reason=f"retries_exceeded:{self._tft_retries}")

    def _is_tft_disabled(self) -> bool:
        """
        TFT is only disabled if TFT_FORCE_DISABLE=true env var is set.
        Filesystem flags are informational only and auto-cleaned on startup.
        """
        force_disable = os.getenv("TFT_FORCE_DISABLE", "").strip().lower() in ("true", "1", "yes")
        if force_disable:
            if not self._tft_disabled_notice_emitted:
                logger.warning(
                    "TFT disabled by TFT_FORCE_DISABLE env var for {}.",
                    self._symbol_slug,
                )
                self._tft_disabled_notice_emitted = True
                self._tft_disabled = True
            return True
        # Auto-clean stale filesystem flags left by the monitor
        if self._tft_disabled and not force_disable:
            self._tft_disabled = False
            logger.info("TFT re-enabled (no TFT_FORCE_DISABLE env var set) for {}", self._symbol_slug)
        return False

    def _disable_tft_permanently(self, reason: str) -> None:
        """Switch to XGB fallback mode but do NOT write a filesystem disable flag.
        The engine retries after a backoff interval to avoid model-load thrashing."""
        self._next_tft_retry_monotonic = max(
            self._next_tft_retry_monotonic,
            time.monotonic() + float(self._tft_retry_backoff_seconds),
        )
        retry_seconds = max(0, int(self._next_tft_retry_monotonic - time.monotonic()))
        logger.warning(
            "TFT model load failed for {} ({}). Switching to DEGRADED_FALLBACK mode. "
            "Next retry in ~{}s.",
            self._symbol_slug,
            reason,
            retry_seconds,
        )
        self._apply_xgb_fallback_mode()

    def _apply_xgb_fallback_mode(self) -> None:
        try:
            self.predictor.model = None
            self.predictor.model_version = "tft_disabled"
        except Exception as exc:
            logger.debug("Could not clear predictor model while entering fallback: {}", exc)
        try:
            self.decision.update_thresholds(
                {
                    "tft_weight": 0.05,
                    "xgb_weight": 0.80,
                    "ppo_weight": 0.15,
                }
            )
            self._save_engine_state("tft_disabled", True)
        except Exception as exc:
            logger.warning("Could not apply fallback agent weights: {}", exc)

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
        for file_path in data_files:
            try:
                df = pd.read_parquet(file_path)
                if "pair" not in df.columns:
                    df["pair"] = file_path.stem.replace(f"_{tf}", "")
                all_dfs.append(df)
            except Exception as exc:
                logger.error(f"Failed reading training file {file_path}: {exc}")

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
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to save auto-training model metrics: {exc}")
        finally:
            session.close()

        logger.info(
            f"Auto-training complete: {version} (val_loss={metrics.get('validation_loss')})"
        )
        register_model_version(
            model_type="tft",
            version=version,
            path=str(Path("saved_models") / version),
            model_metadata=metrics,
            activate=True,
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

        _ = pairs
        symbol = ACTIVE_SYMBOL
        logger.info(f"Bootstrap universe locked to {ACTIVE_SYMBOL}")
        try:
            df = self.fetcher.fetch_history(symbol, timeframe, months)
            if df.empty:
                logger.warning(f"Bootstrap fetch: no data for {symbol} {timeframe}")
                return

            df["pair"] = symbol
            feat_df = compute_features(df, pd.DataFrame())
            out = data_dir / f"{symbol}_{timeframe}.parquet"
            feat_df.to_parquet(out, index=False)
            logger.info(f"Saved bootstrap data: {out}")
        except Exception as exc:
            logger.error(f"Bootstrap fetch failed for {symbol}: {exc}")

    def _should_run_signal_cycle(
        self,
        now_dt: datetime,
        now_mono: float,
        last_signal_time: float,
    ) -> tuple[bool, float]:
        signal_interval = int(getattr(self, "_signal_interval", getattr(self, "_cycle_interval", 300)))
        return self._should_run_timed_cycle(
            key="primary_signal",
            interval_seconds=signal_interval,
            now_dt=now_dt,
            now_mono=now_mono,
            last_run_time=last_signal_time,
        )

    def _should_run_timed_cycle(
        self,
        *,
        key: str,
        interval_seconds: int,
        now_dt: datetime,
        now_mono: float,
        last_run_time: float,
    ) -> tuple[bool, float]:
        interval = max(1, int(interval_seconds))
        if not hasattr(self, "_cycle_buckets") or not isinstance(getattr(self, "_cycle_buckets"), dict):
            self._cycle_buckets = {}
        if self._align_signal_to_interval:
            bucket = int(now_dt.timestamp() // interval)
            if key == "primary_signal" and hasattr(self, "_last_signal_bucket") and self._last_signal_bucket is not None:
                last_bucket = int(self._last_signal_bucket)
            else:
                last_bucket = self._cycle_buckets.get(key)
            if last_bucket is None or bucket > last_bucket:
                self._cycle_buckets[key] = bucket
                if key == "primary_signal":
                    self._last_signal_bucket = bucket
                return True, now_mono
            return False, last_run_time

        if now_mono - last_run_time >= interval:
            return True, now_mono
        return False, last_run_time

    def _market_data_cycle(self) -> None:
        try:
            ticker = self.fetcher.get_ticker(ACTIVE_SYMBOL)
            orderbook = self.fetcher.get_orderbook(ACTIVE_SYMBOL, depth=20)
            bid = float(ticker.get("best_bid") or ticker.get("price") or 0.0)
            ask = float(ticker.get("best_ask") or ticker.get("price") or 0.0)
            price = float(ticker.get("price") or 0.0)
            spread_pct = ((ask - bid) / price) if price > 0 else 0.0
            bid_volume = float(sum(float(level[1]) for level in orderbook.get("bids", []) if len(level) >= 2))
            ask_volume = float(sum(float(level[1]) for level in orderbook.get("asks", []) if len(level) >= 2))
            imbalance = ((bid_volume - ask_volume) / (bid_volume + ask_volume)) if (bid_volume + ask_volume) > 0 else 0.0
            market_status = self.fetcher.get_market_data_status(ACTIVE_SYMBOL)
            self._market_data_status = market_status

            snapshot = {
                "symbol": ACTIVE_SYMBOL,
                "timestamp": datetime.utcnow().isoformat(),
                "price": price,
                "best_bid": bid,
                "best_ask": ask,
                "spread_pct": spread_pct,
                "volume_imbalance": imbalance,
                "source": str(ticker.get("source", "unknown")),
                "ticker_source": str(market_status.get("ticker_source", ticker.get("source", "unknown"))),
                "orderbook_source": str(market_status.get("orderbook_source", orderbook.get("source", "unknown"))),
                "market_data_source": str(market_status.get("market_data_source", "public_ticker")),
                "synthetic_active": bool(market_status.get("synthetic_active", False)),
            }
            self._save_engine_state("market_snapshot", snapshot)
            self._save_engine_state("market_data_status", market_status)
            logger.bind(event="MARKET_DATA_UPDATE", **snapshot).info("MARKET_DATA_UPDATE")
            publish_event("MARKET_DATA_UPDATE", snapshot)
        except Exception as exc:
            logger.warning("Market data cycle failed for {}: {}", ACTIVE_SYMBOL, exc)

    def _restore_runtime_state(self) -> None:
        session = get_session()
        try:
            open_rows = (
                session.query(Trade)
                .filter(Trade.status == "open")
                .order_by(Trade.entry_time.asc(), Trade.id.asc())
                .all()
            )
            closed_rows = (
                session.query(Trade)
                .filter(Trade.status == "closed")
                .order_by(Trade.exit_time.asc(), Trade.id.asc())
                .all()
            )

            self.open_positions = [str(t.trade_id) for t in open_rows if t.trade_id]
            self.trade_history = [str(t.trade_id) for t in closed_rows[-200:] if t.trade_id]
            self.stats.reset()
            for trade in closed_rows:
                self.stats.record_trade(float(trade.pnl or 0.0), float(trade.r_multiple or 0.0))

            logger.bind(
                event="STATE_RESTORED",
                symbol=ACTIVE_SYMBOL,
                open_positions=len(self.open_positions),
                closed_trades=self.stats.total_trades,
                total_pnl=round(self.stats.total_pnl, 4),
            ).info("STATE_RESTORED")
        except Exception as exc:
            logger.warning("Runtime state reconstruction failed: {}", exc)
        finally:
            session.close()

    def _persist_runtime_snapshots(self) -> None:
        session = get_session()
        try:
            balance = 0.0
            realized = 0.0
            unrealized = 0.0
            exposure = 0.0
            open_count = 0
            positions: Dict[str, Any] = {}

            if hasattr(self.executor, "get_metrics_snapshot"):
                snap = self.executor.get_metrics_snapshot()
                balance = float(snap.get("balance", 0.0) or 0.0)
                realized = float(snap.get("realized_pnl", 0.0) or 0.0)
                unrealized = float(snap.get("unrealized_pnl", 0.0) or 0.0)
                positions = snap.get("positions", {}) or {}
            else:
                balance = float(self.executor.get_balance() or 0.0)
                positions = self.executor.get_positions() or {}

            for symbol, payload in positions.items():
                if isinstance(payload, dict):
                    qty = float(payload.get("quantity", 0.0) or 0.0)
                    entry_ref = float(payload.get("avg_entry_price", 0.0) or 0.0)
                else:
                    qty = 0.0
                    entry_ref = 0.0
                if abs(qty) <= 1e-12:
                    continue
                open_count += 1
                try:
                    mark = float(self.fetcher.get_ticker(symbol).get("price") or 0.0)
                except Exception:
                    mark = entry_ref
                exposure += abs(qty * mark)

            equity = float(balance + unrealized)

            session.add(
                EquityHistory(
                    symbol=ACTIVE_SYMBOL,
                    mode=self.mode,
                    balance=float(balance),
                    realized_pnl=float(realized),
                    unrealized_pnl=float(unrealized),
                    equity=float(equity),
                    exposure=float(exposure),
                    open_positions=int(open_count),
                )
            )

            metrics = PerformanceTracker(ACTIVE_SYMBOL).compute_metrics()
            session.add(
                MetricSnapshot(
                    symbol=ACTIVE_SYMBOL,
                    sharpe=float(metrics.get("sharpe_ratio", 0.0) or 0.0),
                    sortino=float(metrics.get("sortino_ratio", 0.0) or 0.0),
                    max_drawdown=float(metrics.get("max_drawdown", 0.0) or 0.0),
                    win_rate=float(metrics.get("win_rate", 0.0) or 0.0),
                    profit_factor=float(metrics.get("profit_factor", 0.0) or 0.0),
                    average_trade=float(metrics.get("average_trade", 0.0) or 0.0),
                    exposure=float(metrics.get("exposure_pct", 0.0) or 0.0),
                    equity=float(equity),
                    rolling_volatility=float(metrics.get("rolling_volatility", 0.0) or 0.0),
                    total_trades=int(metrics.get("total_trades", 0) or 0),
                    winning_trades=int(metrics.get("win_count", 0) or 0),
                    losing_trades=int(metrics.get("loss_count", 0) or 0),
                )
            )

            stats_row = session.query(Statistics).order_by(Statistics.id.asc()).first()
            if stats_row is None:
                stats_row = Statistics()
                session.add(stats_row)
            stats_row.total_trades = int(self.stats.total_trades)
            stats_row.wins = int(self.stats.wins)
            stats_row.losses = int(self.stats.losses)
            stats_row.win_rate = float(self.stats.win_rate)
            stats_row.avg_r = float(self.stats.avg_r)
            stats_row.total_pnl = float(self.stats.total_pnl)
            stats_row.unrealized_pnl = float(unrealized)

            session.commit()
        except Exception as exc:
            session.rollback()
            logger.debug("Snapshot persistence skipped: {}", exc)
        finally:
            session.close()

    def _watchdog_check(self, now_mono: float) -> None:
        if self.shutdown_controller.should_stop():
            return

        stall_seconds = now_mono - self._last_signal_cycle_monotonic
        threshold_seconds = float(self._cycle_interval * self._watchdog_missed_cycles)
        if stall_seconds <= threshold_seconds:
            return
        if now_mono - self._watchdog_last_warning_monotonic < max(self._monitor_interval, 5):
            return

        self._watchdog_last_warning_monotonic = now_mono
        warning_payload = {
            "stalled_for_seconds": round(stall_seconds, 1),
            "threshold_seconds": round(threshold_seconds, 1),
            "last_reason": self._last_cycle_reason,
            "cycle_interval_seconds": self._cycle_interval,
        }
        logger.bind(event="DECISION_WATCHDOG", **warning_payload).warning("DECISION_WATCHDOG")
        self._save_engine_state("watchdog_last_warning", warning_payload)

    def _register_no_trade_reason(
        self,
        reason_code: str,
        detail: Optional[str] = None,
        signal: Optional[object] = None,
        persist_event: bool = True,
    ) -> None:
        reason = str(reason_code).strip() or "unknown_no_trade_reason"
        self._last_cycle_reason = reason
        self._no_trade_reason_counts[reason] = self._no_trade_reason_counts.get(reason, 0) + 1

        detail_text = str(detail).strip() if detail else ""
        logger.bind(
            event="NO_TRADE",
            reason_code=reason,
            detail=detail_text or None,
            count=self._no_trade_reason_counts[reason],
            pair=getattr(signal, "pair", None),
        ).info("NO_TRADE")

        if not persist_event:
            return

        session = get_session()
        try:
            event = DecisionEvent(
                timestamp=datetime.utcnow(),
                mode=settings.trading.trading_mode.upper(),
                status="no_trade",
                reason=reason,
                candidates_evaluated=0,
                candidates_valid=1 if signal is not None else 0,
                best_pair=getattr(signal, "pair", None),
                best_score=_finite_or_none(getattr(signal, "ai_score", None)),
                best_ai_score=_finite_or_none(getattr(signal, "final_ai_score", None)),
                best_confidence=_finite_or_none(getattr(signal, "confidence", None)),
                best_prob_up=_finite_or_none(getattr(signal, "prob_up", None)),
                best_prob_down=_finite_or_none(getattr(signal, "prob_down", None)),
                regime=getattr(signal, "market_regime", None),
                volatility_regime=getattr(signal, "volatility_regime", None),
                adaptive_threshold=_finite_or_none(getattr(signal, "adaptive_threshold", None)),
                top_candidates_json=(
                    [{
                        "pair": getattr(signal, "pair", None),
                        "score": _finite_or_none(getattr(signal, "ai_score", None), 4),
                        "prob_up": _finite_or_none(getattr(signal, "prob_up", None), 4),
                        "prob_down": _finite_or_none(getattr(signal, "prob_down", None), 4),
                        "confidence": _finite_or_none(getattr(signal, "confidence", None), 4),
                        "detail": detail_text or None,
                    }]
                    if signal is not None
                    else [{"detail": detail_text or None}]
                ),
            )
            session.add(event)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning("Could not persist no_trade decision event ({}): {}", reason, exc)
        finally:
            session.close()

    def _increment_engine_counter(self, key: str, amount: int = 1) -> None:
        session = get_session()
        try:
            state = session.query(EngineState).filter(EngineState.key == key).first()
            current = 0
            if state and isinstance(state.value, dict):
                try:
                    current = int(state.value.get("value", 0) or 0)
                except Exception:
                    current = 0
            payload = {"value": int(current + int(amount))}
            if state:
                state.value = payload
                state.updated_at = datetime.utcnow()
            else:
                session.add(EngineState(key=key, value=payload))
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.debug("Could not increment counter {}: {}", key, exc)
        finally:
            session.close()

    def _increment_rejection_reason(self, reason: str) -> None:
        reason_key = str(reason or "unknown").strip() or "unknown"
        session = get_session()
        try:
            state = session.query(EngineState).filter(EngineState.key == "trade_rejection_reasons").first()
            data: Dict[str, int] = {}
            if state and isinstance(state.value, dict):
                raw = state.value.get("value", {})
                if isinstance(raw, dict):
                    for key, value in raw.items():
                        try:
                            data[str(key)] = int(value)
                        except Exception:
                            continue
            data[reason_key] = int(data.get(reason_key, 0) + 1)
            payload = {"value": data}
            if state:
                state.value = payload
                state.updated_at = datetime.utcnow()
            else:
                session.add(EngineState(key="trade_rejection_reasons", value=payload))
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.debug("Could not update rejection reason counter {}: {}", reason_key, exc)
        finally:
            session.close()

    def _record_trade_rejection(self, reason: str) -> None:
        self._increment_engine_counter("trade_rejection_count", 1)
        self._increment_rejection_reason(reason)

    def _mark_cycle_started(self, cycle_label: str) -> None:
        now_iso = datetime.utcnow().isoformat()
        self._increment_engine_counter("decision_cycle_count", 1)
        self._save_engine_state("last_decision_timestamp", now_iso)
        self._save_engine_state("last_cycle_terminal_state", "IN_PROGRESS")
        self._save_engine_state("last_cycle_reason", f"cycle_{cycle_label}_started")
        active_model_name = str(self.predictor.model_version or "xgb_meta_fallback")
        self._save_engine_state("active_model_name", active_model_name)

    def _mark_cycle_terminal(self, terminal_state: str, reason: str) -> None:
        now_iso = datetime.utcnow().isoformat()
        normalized_terminal = str(terminal_state or "TRADE_SKIPPED_WITH_REASON")
        normalized_reason = str(reason or "unspecified")
        self._last_cycle_reason = normalized_reason
        self._save_engine_state("last_cycle_terminal_state", normalized_terminal)
        self._save_engine_state("last_cycle_reason", normalized_reason)
        self._save_engine_state("last_decision_timestamp", now_iso)
        if normalized_terminal == "TRADE_EXECUTED":
            self._save_engine_state("last_trade_timestamp", now_iso)

    def _signal_cycle(self, cycle_label: str = "1m") -> None:
        """Run one signal generation and execution cycle."""
        self._last_signal_cycle_monotonic = time.monotonic()
        self._mark_cycle_started(cycle_label)
        if self.shutdown_controller.should_stop() or not self._accept_new_trades:
            logger.info("[ENGINE] Shutdown requested; skipping new trade execution")
            self._register_no_trade_reason("shutdown_or_accept_new_trades_disabled", persist_event=False)
            publish_event("RISK_REJECTED", {"symbol": ACTIVE_SYMBOL, "reason": "shutdown", "cycle": cycle_label})
            self._record_trade_rejection("shutdown_or_accept_new_trades_disabled")
            self._mark_cycle_terminal("ENGINE_HALTED", "shutdown_or_accept_new_trades_disabled")
            return

        if is_safe_mode():
            logger.critical("[ENGINE] SAFE_MODE active; trading disabled until manual reset")
            self._register_no_trade_reason("safe_mode_active")
            publish_event("RISK_REJECTED", {"symbol": ACTIVE_SYMBOL, "reason": "safe_mode", "cycle": cycle_label})
            self._record_trade_rejection("safe_mode_active")
            self._mark_cycle_terminal("ENGINE_HALTED", "safe_mode_active")
            return

        try:
            equity = self.prop_risk.estimate_equity()
            triggered, reason, _ctx = evaluate_and_arm_kill_switch(equity)
        except Exception as exc:
            logger.exception("[ENGINE] Kill-switch evaluation error: {}", exc)
            triggered = False
            reason = ""

        if triggered:
            logger.critical(
                "[ENGINE] Kill-switch triggered ({}). Closing positions and entering SAFE_MODE", reason
            )
            self._register_no_trade_reason("kill_switch_triggered", detail=reason)
            publish_event("RISK_REJECTED", {"symbol": ACTIVE_SYMBOL, "reason": f"kill_switch:{reason}", "cycle": cycle_label})
            try:
                self.safety.pause_trading()
            except Exception:
                logger.exception("[ENGINE] Failed to pause trading after kill-switch")
            try:
                self.monitor.force_close_all()
            except Exception:
                logger.exception("[ENGINE] Failed to force-close positions after kill-switch")
            self._record_trade_rejection("kill_switch_triggered")
            self._mark_cycle_terminal("ENGINE_HALTED", "kill_switch_triggered")
            return

        can_trade, reason = self.safety.can_trade()
        if not can_trade:
            logger.info(f"Cannot trade: {reason}")
            self._register_no_trade_reason("safety_can_trade_blocked", detail=reason)
            publish_event("RISK_REJECTED", {"symbol": ACTIVE_SYMBOL, "reason": str(reason), "cycle": cycle_label})
            self._record_trade_rejection("safety_can_trade_blocked")
            self._mark_cycle_terminal("TRADE_SKIPPED_WITH_REASON", f"safety_can_trade_blocked:{reason}")
            return

        if not settings.trading.trading_enabled:
            self._register_no_trade_reason("trading_disabled")
            publish_event("RISK_REJECTED", {"symbol": ACTIVE_SYMBOL, "reason": "trading_disabled", "cycle": cycle_label})
            self._record_trade_rejection("trading_disabled")
            self._mark_cycle_terminal("TRADE_SKIPPED_WITH_REASON", "trading_disabled")
            return

        with self._inference_scope():
            signal = self.decision.generate_signal()
        if signal is None:
            logger.debug("No valid signal generated")
            # DecisionEngine already persists no_trade for this branch.
            self._register_no_trade_reason("no_signal_from_decision_engine", persist_event=False)
            self._record_trade_rejection("no_signal_from_decision_engine")
            self._mark_cycle_terminal("TRADE_SKIPPED_WITH_REASON", "no_signal_from_decision_engine")
            return

        logger.bind(
            event="SIGNAL_GENERATED",
            symbol=signal.pair,
            side=signal.side,
            confidence=round(float(signal.confidence), 4),
            signal_score=round(float(getattr(signal, "signal_score", 0.0)), 4),
            cycle=cycle_label,
        ).info("SIGNAL_GENERATED")
        publish_event(
            "SIGNAL_GENERATED",
            {
                "symbol": signal.pair,
                "side": signal.side,
                "confidence": float(signal.confidence),
                "signal_score": float(getattr(signal, "signal_score", 0.0)),
                "cycle": cycle_label,
            },
        )
        logger.info(
            f"SIGNAL GENERATED | symbol={signal.pair} | confidence={signal.confidence:.3f} | "
            f"tft={signal.tft_score:.3f} | xgb={signal.xgb_score:.3f} | "
            f"regime={signal.market_regime} | vol_regime={signal.volatility_regime} | "
            f"expected_move={signal.expected_move:.4f} | prob_up={signal.prob_up:.3f} | "
            f"prob_down={signal.prob_down:.3f} | entry={signal.entry_price:.6f} | "
            f"stop={signal.stop_price:.6f} | target={signal.target_price:.6f}"
        )

        if self.safety.check_volatility_circuit_breaker(
            signal.atr / signal.entry_price,
            0.02,
        ):
            logger.warning("Circuit breaker: skipping trade")
            self._register_no_trade_reason("volatility_circuit_breaker", signal=signal)
            publish_event("RISK_REJECTED", {"symbol": signal.pair, "reason": "volatility_circuit_breaker", "cycle": cycle_label})
            self._record_trade_rejection("volatility_circuit_breaker")
            self._mark_cycle_terminal("TRADE_SKIPPED_WITH_REASON", "volatility_circuit_breaker")
            return

        # Never start a new order submission after shutdown was requested.
        if self.shutdown_controller.should_stop() or not self._accept_new_trades:
            logger.info("[ENGINE] Finishing current cycle without opening new trade")
            self._register_no_trade_reason("shutdown_before_execution", signal=signal, persist_event=False)
            publish_event("RISK_REJECTED", {"symbol": signal.pair, "reason": "shutdown_before_execution", "cycle": cycle_label})
            self._record_trade_rejection("shutdown_before_execution")
            self._mark_cycle_terminal("ENGINE_HALTED", "shutdown_before_execution")
            return

        balance = self.executor.get_balance()
        if settings.trading.trading_mode.upper() == "PAPER":
            side_for_sizing = str(getattr(signal, "side", "BUY")).upper()
            cash_balance = max(0.0, float(balance or 0.0))
            balance = cash_balance
            try:
                metrics = self.executor.get_metrics()
                equity_balance = float((metrics or {}).get("equity", cash_balance))
                if math.isfinite(equity_balance):
                    # Side-aware paper sizing:
                    # - BUY uses available cash (avoid repeated rejected BUYs when cash is nearly exhausted).
                    # - SELL uses equity (avoid short proceeds inflating position size).
                    if side_for_sizing == "SELL":
                        balance = max(0.0, equity_balance)
                    else:
                        balance = cash_balance
            except Exception as exc:
                logger.debug("Could not use paper metrics for sizing: {}", exc)
        rl_state = {
            "entry_price": signal.entry_price,
            "current_price": signal.entry_price,
            "quantity": 1.0,
            "rsi": signal.features_snapshot.get("rsi_14", 50.0),
            "ema_20": signal.features_snapshot.get("ema_21", signal.entry_price),
            "volatility": signal.features_snapshot.get("volatility_20", 0.0),
            "unrealized_pnl": 0.0,
            "time_in_trade": 0.0,
        }
        risk_multiplier = self.rl_manager.initial_size(rl_state)
        signal.ppo_score = _normalize_ppo_score(risk_multiplier)

        # Aggressive ensemble blending: TFT 50%, XGB 30%, META/PPO 20%
        tft_conf = float(signal.tft_score)
        xgb_conf = float(signal.xgb_score)
        meta_conf = float(signal.ppo_score)

        weight_snapshot = {"tft_weight": 0.5, "xgb_weight": 0.3, "ppo_weight": 0.2}
        weight_total = 1.0

        base_ai_score = _clamp01(
            tft_conf * 0.5 + xgb_conf * 0.3 + meta_conf * 0.2
        )
        signal.weight_snapshot = weight_snapshot

        logger.info(
            f"CONFIDENCE BLEND | TFT={tft_conf:.3f} | XGB={xgb_conf:.3f} | "
            f"META={meta_conf:.3f} | FINAL={base_ai_score:.3f}"
        )

        governance_decision = self.governance.evaluate(
            signal=signal,
            ppo_size_mult=risk_multiplier,
        )
        signal.governance_approved = bool(governance_decision.approve)
        signal.governance_code = governance_decision.code
        signal.governance_size_mult = float(governance_decision.size_mult)
        signal.governance_conf_adj = float(governance_decision.conf_adj)
        signal.governance_risk_mode = governance_decision.risk_mode
        signal.gov_adjust = float(governance_decision.conf_adj)

        final_ai_score = _clamp01(base_ai_score + governance_decision.conf_adj)
        signal.base_ai_score = base_ai_score
        signal.ai_score = final_ai_score
        signal.final_ai_score = final_ai_score
        signal.confidence = final_ai_score

        logger.info(
            f"DECISION FLOW | symbol={signal.pair} | "
            f"raw_prediction={{prob_up={signal.prob_up:.3f}, prob_down={signal.prob_down:.3f}}} | "
            f"blended_confidence={final_ai_score:.3f} | threshold=0.45 | "
            f"regime={signal.market_regime} | "
            f"position_size_mult={signal.position_size_multiplier:.2f} | "
            f"governance={governance_decision.code}"
        )

        self.governance.record_audit(
            signal=signal,
            ppo_size_mult=risk_multiplier,
            governance_decision=governance_decision,
            final_ai_score=final_ai_score,
        )

        if not governance_decision.approve:
            logger.info(
                f"TRADE SKIPPED | symbol={signal.pair} | reason=governance_rejected | "
                f"code={governance_decision.code} | risk_mode={governance_decision.risk_mode} | "
                f"confidence={final_ai_score:.3f}"
            )
            self._register_no_trade_reason(
                "governance_rejected",
                detail=f"{governance_decision.code}:{governance_decision.risk_mode}",
                signal=signal,
            )
            publish_event(
                "RISK_REJECTED",
                {
                    "symbol": signal.pair,
                    "reason": f"governance:{governance_decision.code}",
                    "cycle": cycle_label,
                },
            )
            self._record_trade_rejection("governance_rejected")
            self._mark_cycle_terminal("TRADE_SKIPPED_WITH_REASON", f"governance_rejected:{governance_decision.code}")
            return

        prop_ok, prop_reason = self.prop_risk.can_open_new_trade(
            execution_in_progress=self._trade_execution_in_progress
        )
        if not prop_ok:
            logger.warning(
                f"TRADE SKIPPED | symbol={signal.pair} | reason=prop_risk_blocked | "
                f"detail={prop_reason} | confidence={final_ai_score:.3f}"
            )
            self._register_no_trade_reason("prop_risk_blocked", detail=prop_reason, signal=signal)
            publish_event("RISK_REJECTED", {"symbol": signal.pair, "reason": f"prop_risk:{prop_reason}", "cycle": cycle_label})
            self._record_trade_rejection("prop_risk_blocked")
            self._mark_cycle_terminal("TRADE_SKIPPED_WITH_REASON", f"prop_risk_blocked:{prop_reason}")
            return

        final_risk_multiplier = max(
            0.1,
            min(2.0, float(risk_multiplier * governance_decision.size_mult)),
        )

        self._increment_engine_counter("trade_attempt_count", 1)
        with self._trade_execution_scope():
            trade_id = self.executor.execute_signal(
                signal,
                balance,
                risk_multiplier=final_risk_multiplier,
            )

        if trade_id:
            self.open_positions.append(trade_id)
            logger.bind(
                event="TRADE_SUBMITTED",
                trade_id=trade_id,
                symbol=signal.pair,
                side=signal.side,
                confidence=round(signal.confidence, 4),
                engine_state=self._engine_state,
            ).info("TRADE_SUBMITTED")
            self._last_cycle_reason = "trade_opened"
            self._record_trade_opened_event(signal)
            logger.bind(
                event="TRADE_OPENED",
                trade_id=trade_id,
                symbol=signal.pair,
                side=signal.side,
                cycle=cycle_label,
            ).info("TRADE_OPENED")
            self._mark_cycle_terminal("TRADE_EXECUTED", "trade_opened")
        else:
            logger.bind(
                event="TRADE_REJECTED",
                symbol=signal.pair,
                side=signal.side,
                confidence=round(signal.confidence, 4),
                reason="execution_rejected_or_failed",
                engine_state=self._engine_state,
            ).warning("TRADE_REJECTED")
            self._register_no_trade_reason("execution_rejected_or_failed", signal=signal)
            publish_event("RISK_REJECTED", {"symbol": signal.pair, "reason": "execution_rejected_or_failed", "cycle": cycle_label})
            self._record_trade_rejection("execution_rejected_or_failed")
            self._mark_cycle_terminal("TRADE_SKIPPED_WITH_REASON", "execution_rejected_or_failed")

    def _record_trade_opened_event(self, signal) -> None:
        """Persist a decision event when a trade IS opened."""
        from data.database import DecisionEvent

        session = get_session()
        try:
            event = DecisionEvent(
                timestamp=datetime.utcnow(),
                mode=settings.trading.trading_mode.upper(),
                status="trade_opened",
                reason=f"{signal.side} signal for {signal.pair}",
                candidates_evaluated=0,
                candidates_valid=1,
                best_pair=signal.pair,
                best_score=_finite_or_none(signal.ai_score),
                best_ai_score=_finite_or_none(signal.final_ai_score),
                best_confidence=_finite_or_none(signal.confidence),
                best_prob_up=_finite_or_none(signal.prob_up),
                best_prob_down=_finite_or_none(signal.prob_down),
                regime=signal.market_regime,
                volatility_regime=signal.volatility_regime,
                adaptive_threshold=_finite_or_none(getattr(signal, "adaptive_threshold", 0.0)),
                top_candidates_json=[{
                    "pair": signal.pair,
                    "score": _finite_or_none(signal.ai_score, 4),
                    "prob_up": _finite_or_none(signal.prob_up, 4),
                    "prob_down": _finite_or_none(signal.prob_down, 4),
                    "confidence": _finite_or_none(signal.confidence, 4),
                    "side": signal.side,
                }],
            )
            session.add(event)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.warning(f"Could not persist trade_opened decision event: {e}")
        finally:
            session.close()

    def _feedback_cycle(self) -> None:
        """Run feedback analysis and apply adjustments."""
        adjustments = self.feedback.compute_batch_adjustments()
        if adjustments:
            self.decision.update_thresholds(adjustments)
            logger.info(f"Applied threshold adjustments: {adjustments}")

    def update_risk_metrics(self) -> None:
        """Recompute and persist rolling Sharpe/Sortino/max-drawdown snapshot."""
        try:
            equity = self.prop_risk.estimate_equity()
            update_risk_metrics_snapshot(equity_override=equity)
        except Exception as exc:
            logger.error(f"Risk metric update failed: {exc}")

    def _shutdown(self) -> None:
        """Clean shutdown with deterministic sequence and timeout protection."""
        self.shutdown_controller.initiate(reason="engine_stop")
        self._accept_new_trades = False
        self._health_ready = False
        self.reconciliation.stop()

        self._save_engine_state("running", False)
        self._save_engine_state("accept_new_trades", False)
        self._save_engine_state("thresholds", self.decision.get_current_thresholds())

        deadline = time.monotonic() + 30
        logger.info("[ENGINE] Cleaning resources")

        try:
            self.stop_trading(deadline)
            self.close_positions_if_configured()
            self.close_db()
            self.close_redis()
            self.release_gpu(deadline)
        except Exception as exc:
            logger.error(f"[ENGINE] Shutdown sequence failed: {exc}")
        finally:
            self.governance.shutdown()
            self.instance_lock.release()
            if self._health_server is not None:
                try:
                    self._health_server.stop()
                except Exception:
                    logger.exception("[HEALTH] Failed to stop health server")

        if time.monotonic() > deadline:
            logger.error("[ENGINE] Shutdown exceeded 30 seconds. Forcing exit.")
            os._exit(1)  # noqa: S404 - explicit hard-exit path on timeout

        logger.info("[ENGINE] Shutdown complete")

    def stop_trading(self, deadline: float) -> None:
        """Stop opening new positions; optionally keep monitoring live positions."""
        logger.info("[ENGINE] Finishing current cycle")
        self._accept_new_trades = False
        self._wait_for_inflight_work(deadline)

        mode = settings.trading.trading_mode.upper()
        if mode == "PAPER":
            logger.info("[ENGINE] PAPER mode: safe immediate stop.")
            return

        open_count = self._count_open_trades()
        if open_count <= 0:
            return

        logger.info(f"[ENGINE] LIVE mode: monitoring {open_count} open trade(s) before shutdown.")
        original_rl_manager = self.monitor.rl_manager
        self.monitor.rl_manager = None
        try:
            while open_count > 0 and time.monotonic() < deadline:
                try:
                    with self._inference_scope():
                        self.monitor.monitor_cycle()
                except Exception as exc:
                    logger.error(f"[ENGINE] Shutdown monitor cycle failed: {exc}")
                open_count = self._count_open_trades()
                if open_count > 0:
                    self._sleep_interruptible(1, break_on_shutdown=False)

            if open_count > 0:
                logger.warning(
                    f"[ENGINE] Shutdown timeout reached with {open_count} open trade(s) still active."
                )
        finally:
            self.monitor.rl_manager = original_rl_manager

    def close_positions_if_configured(self) -> None:
        """Optionally force-close open positions during shutdown."""
        if not self._env_bool("CLOSE_POSITIONS_ON_SHUTDOWN", False):
            return

        if settings.trading.trading_mode.upper() != "LIVE":
            logger.info("[ENGINE] Skipping forced position close in PAPER mode")
            return

        logger.warning("[ENGINE] Force-closing open positions due to CLOSE_POSITIONS_ON_SHUTDOWN")
        try:
            self.monitor.force_close_all()
        except Exception as exc:
            logger.error(f"[ENGINE] Force-close failed: {exc}")

    def close_db(self) -> None:
        """Close active sessions and dispose SQLAlchemy engine."""
        logger.info("[ENGINE] Cleaning resources: database")
        try:
            session = get_session()
            try:
                if session.in_transaction():
                    session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
        except Exception as exc:
            logger.error(f"[ENGINE] Database session close error: {exc}")
        finally:
            try:
                remove_session()
            except Exception as exc:
                logger.error(f"[ENGINE] Scoped session cleanup error: {exc}")
            try:
                dispose_engine()
            except Exception as exc:
                logger.error(f"[ENGINE] Engine dispose error: {exc}")

    def close_redis(self) -> None:
        """Close redis-like clients if present."""
        logger.info("[ENGINE] Cleaning resources: redis")
        closed_any = False
        for client in self._iter_redis_clients():
            try:
                if hasattr(client, "close"):
                    client.close()
                    closed_any = True
                elif hasattr(client, "connection_pool"):
                    client.connection_pool.disconnect()
                    closed_any = True
            except Exception as exc:
                logger.error(f"[ENGINE] Redis close error: {exc}")
        if closed_any:
            logger.info("[ENGINE] Redis connections closed")

    def release_gpu(self, deadline: float) -> None:
        """Release model references and clear CUDA cache."""
        logger.info("[ENGINE] Cleaning resources: model and GPU")

        wait_logged = False
        while self._is_inference_active() and time.monotonic() < deadline:
            if not wait_logged:
                logger.info("[ENGINE] Waiting for active inference to complete")
                wait_logged = True
            self._sleep_interruptible(1, break_on_shutdown=False)

        model_ref = getattr(self.predictor, "model", None)
        if model_ref is not None:
            self.predictor.model = None
            del model_ref

        gc.collect()

        try:
            import torch
        except Exception:
            return

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.error(f"[ENGINE] GPU cleanup error: {exc}")

    def _count_open_trades(self) -> int:
        from data.database import Trade

        session = get_session()
        try:
            return int(session.query(Trade).filter(Trade.status == "open").count())
        finally:
            session.close()

    def _wait_for_inflight_work(self, deadline: float) -> None:
        wait_logged = False
        while self._has_inflight_work() and time.monotonic() < deadline:
            if not wait_logged:
                logger.info("[ENGINE] Waiting for in-flight execution to finish")
                wait_logged = True
            self._sleep_interruptible(1, break_on_shutdown=False)

    def _has_inflight_work(self) -> bool:
        with self._state_lock:
            return (
                self._cycle_in_progress
                or self._trade_execution_in_progress
                or self._inference_in_progress
            )

    def _is_inference_active(self) -> bool:
        with self._state_lock:
            return self._inference_in_progress

    def _sleep_interruptible(self, seconds: int, break_on_shutdown: bool = True) -> None:
        whole_seconds = max(0, int(seconds))
        for _ in range(whole_seconds):
            if break_on_shutdown and self.shutdown_controller.should_stop():
                return
            time.sleep(1)

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name, "")
        if not raw.strip():
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return int(default)
        try:
            return int(raw)
        except ValueError:
            return int(default)

    def _iter_redis_clients(self) -> Iterator[object]:
        seen = set()
        components = (
            self,
            self.fetcher,
            self.executor,
            self.monitor,
            self.decision,
            self.feedback,
            self.safety,
        )
        attr_names = ("redis", "redis_client", "_redis", "_redis_client", "cache")
        for component in components:
            for attr_name in attr_names:
                client = getattr(component, attr_name, None)
                if client is None:
                    continue
                key = id(client)
                if key in seen:
                    continue
                module_name = getattr(client.__class__, "__module__", "").lower()
                if "redis" not in module_name and "redis" not in attr_name:
                    continue
                seen.add(key)
                yield client

    @contextmanager
    def _cycle_scope(self) -> Iterator[None]:
        with self._state_lock:
            self._cycle_in_progress = True
        try:
            yield
        finally:
            with self._state_lock:
                self._cycle_in_progress = False

    @contextmanager
    def _inference_scope(self) -> Iterator[None]:
        with self._state_lock:
            self._inference_in_progress = True
        try:
            yield
        finally:
            with self._state_lock:
                self._inference_in_progress = False

    @contextmanager
    def _trade_execution_scope(self) -> Iterator[None]:
        with self._state_lock:
            self._trade_execution_in_progress = True
        try:
            yield
        finally:
            with self._state_lock:
                self._trade_execution_in_progress = False

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
        except Exception as exc:
            session.rollback()
            logger.error(f"Error saving engine state: {exc}")
        finally:
            session.close()


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="TFT AI Trading Engine")
    parser.add_argument("--model", type=str, help="Model version to load")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Paper trading mode (no real orders)",
    )
    args = parser.parse_args()

    if args.dry_run and settings.trading.trading_mode.upper() != "PAPER":
        import config.settings as cfg_mod

        os.environ["TRADING_MODE"] = "PAPER"
        cfg_mod.settings = cfg_mod.Settings()
        globals()["settings"] = cfg_mod.settings
        logger.info("Paper trading mode enabled via --dry-run flag")

    if (
        settings.trading.trading_mode.upper() == "LIVE"
        and settings.trading.allow_live_trading.strip().lower() != "true"
    ):
        raise RuntimeError("LIVE trading blocked. Set ALLOW_LIVE_TRADING=true explicitly.")

    engine = TradingEngine()
    engine.start(model_version=args.model)


if __name__ == "__main__":
    main()
