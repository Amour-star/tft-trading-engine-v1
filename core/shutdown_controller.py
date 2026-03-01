"""Global shutdown coordination and process signal handling."""
from __future__ import annotations

import os
import signal
import threading
from types import FrameType
from typing import Optional

from loguru import logger


class ShutdownController:
    """Tracks process shutdown intent and handles SIGINT/SIGTERM deterministically."""

    def __init__(self) -> None:
        self.shutdown_requested: bool = False
        self._force_exit_requested: bool = False
        self._signal_count: int = 0
        self._signals_bound: bool = False
        self._lock = threading.Lock()

    def bind_signals(self) -> None:
        """Register SIGINT/SIGTERM handlers once."""
        with self._lock:
            if self._signals_bound:
                return

            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
            self._signals_bound = True

    def should_stop(self) -> bool:
        return self.shutdown_requested

    def force_exit_requested(self) -> bool:
        return self._force_exit_requested

    def initiate(self, reason: Optional[str] = None) -> bool:
        """
        Request graceful shutdown.

        Returns True if this call initiated shutdown, False if already requested.
        """
        with self._lock:
            if self.shutdown_requested:
                return False
            self.shutdown_requested = True

        if reason:
            logger.warning(f"[ENGINE] Shutdown requested ({reason})")
        else:
            logger.warning("[ENGINE] Shutdown requested")
        logger.warning("[ENGINE] Graceful shutdown initiated...")
        return True

    def _handle_signal(self, signum: int, _frame: Optional[FrameType]) -> None:
        """First signal triggers graceful stop, second signal forces immediate exit."""
        signal_name = self._signal_name(signum)

        with self._lock:
            self._signal_count += 1
            count = self._signal_count

        if count == 1:
            self.initiate(reason=signal_name)
            return

        with self._lock:
            if self._force_exit_requested:
                return
            self._force_exit_requested = True

        logger.error(f"[ENGINE] Second shutdown signal received ({signal_name}). Forcing exit.")
        os._exit(1)  # noqa: S404 - explicit hard-exit path on repeated signal

    @staticmethod
    def _signal_name(signum: int) -> str:
        try:
            return signal.Signals(signum).name
        except Exception:
            return str(signum)


shutdown_controller = ShutdownController()

