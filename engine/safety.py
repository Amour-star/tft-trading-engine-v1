"""
Safety layer: circuit breakers, max loss limits, risk controls.
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Optional

from loguru import logger
from sqlalchemy import func

from config.settings import settings
from data.database import get_session, Trade, DailyStats, EngineState


class SafetyManager:
    """Monitors and enforces trading safety limits."""

    def __init__(self) -> None:
        self.cfg = settings.safety
        self._paused: bool = False
        self._killed: bool = False
        self._consecutive_losses: int = 0
        self._daily_pnl: float = 0.0
        self._daily_trade_count: int = 0
        self._last_check_date: Optional[date] = None

    def can_trade(self) -> tuple[bool, str]:
        if self._killed:
            return False, "Emergency kill switch activated"
        if self._paused:
            return False, "Trading is paused"

        self._refresh_daily_stats()

        balance = self._get_starting_balance()
        if balance > 0 and self._daily_pnl < 0:
            if abs(self._daily_pnl) / balance > self.cfg.max_daily_loss_pct:
                self._paused = True
                return False, f"Daily loss limit exceeded ({self.cfg.max_daily_loss_pct:.1%})"

        if self._consecutive_losses >= self.cfg.max_consecutive_losses:
            return False, f"Max consecutive losses reached ({self._consecutive_losses})"

        session = get_session()
        try:
            open_trades = session.query(Trade).filter(Trade.status == "open").count()
            if open_trades >= settings.trading.max_open_trades:
                return False, f"Max open trades reached ({open_trades})"
        finally:
            session.close()

        return True, "All checks passed"

    def record_trade_result(self, pnl: float, trade_id: str) -> None:
        self._daily_pnl += pnl
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self._update_daily_stats(pnl)

    def check_volatility_circuit_breaker(self, current_vol: float, baseline_vol: float) -> bool:
        if baseline_vol <= 0:
            return False
        ratio = current_vol / baseline_vol
        if ratio > self.cfg.circuit_breaker_volatility_multiplier:
            logger.warning(f"Volatility circuit breaker: {ratio:.2f}x baseline")
            return True
        return False

    def check_api_stability(self, consecutive_errors: int) -> bool:
        return consecutive_errors < self.cfg.api_error_threshold

    def pause_trading(self) -> None:
        self._paused = True
        self._save_state("paused", True)
        logger.warning("Trading PAUSED")

    def resume_trading(self) -> None:
        self._paused = False
        self._save_state("paused", False)
        logger.info("Trading RESUMED")

    def emergency_kill(self) -> None:
        self._killed = True
        self._save_state("killed", True)
        logger.critical("EMERGENCY KILL SWITCH ACTIVATED")

    def reset_kill_switch(self) -> None:
        self._killed = False
        self._save_state("killed", False)

    def reset_consecutive_losses(self) -> None:
        self._consecutive_losses = 0

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_killed(self) -> bool:
        return self._killed

    def get_status(self) -> dict:
        return {
            "paused": self._paused,
            "killed": self._killed,
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl": self._daily_pnl,
            "daily_trade_count": self._daily_trade_count,
            "max_daily_loss_pct": self.cfg.max_daily_loss_pct,
            "max_consecutive_losses": self.cfg.max_consecutive_losses,
        }

    def load_state(self) -> None:
        session = get_session()
        try:
            for key in ["paused", "killed"]:
                state = session.query(EngineState).filter(EngineState.key == key).first()
                if state:
                    setattr(self, f"_{key}", bool(state.value.get("value", False)))
            self._refresh_daily_stats()
        finally:
            session.close()

    def _save_state(self, key: str, value: bool) -> None:
        session = get_session()
        try:
            state = session.query(EngineState).filter(EngineState.key == key).first()
            if state:
                state.value = {"value": value}
            else:
                session.add(EngineState(key=key, value={"value": value}))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving state {key}: {e}")
        finally:
            session.close()

    def _refresh_daily_stats(self) -> None:
        today = date.today()
        if self._last_check_date == today:
            return
        session = get_session()
        try:
            today_start = datetime.combine(today, datetime.min.time())
            result = session.query(
                func.sum(Trade.pnl), func.count(Trade.id),
            ).filter(Trade.exit_time >= today_start, Trade.status == "closed").first()
            self._daily_pnl = float(result[0] or 0)
            self._daily_trade_count = int(result[1] or 0)

            recent = (
                session.query(Trade).filter(Trade.status == "closed")
                .order_by(Trade.exit_time.desc())
                .limit(self.cfg.max_consecutive_losses + 1).all()
            )
            self._consecutive_losses = 0
            for t in recent:
                if t.pnl is not None and t.pnl < 0:
                    self._consecutive_losses += 1
                else:
                    break
            self._last_check_date = today
        finally:
            session.close()

    def _get_starting_balance(self) -> float:
        session = get_session()
        try:
            today = date.today()
            stats = session.query(DailyStats).filter(DailyStats.date == today).first()
            return stats.balance_start if stats and stats.balance_start else 10000.0
        finally:
            session.close()

    def _update_daily_stats(self, pnl: float) -> None:
        session = get_session()
        try:
            today = date.today()
            stats = session.query(DailyStats).filter(DailyStats.date == today).first()
            if not stats:
                stats = DailyStats(date=today, total_pnl=0, total_trades=0, winning_trades=0, losing_trades=0)
                session.add(stats)
            stats.total_trades += 1
            stats.total_pnl += pnl
            if pnl >= 0:
                stats.winning_trades += 1
            else:
                stats.losing_trades += 1
            stats.consecutive_losses = self._consecutive_losses
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating daily stats: {e}")
        finally:
            session.close()
