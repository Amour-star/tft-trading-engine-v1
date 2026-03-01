"""
Safety layer: circuit breakers, max loss limits, risk controls.
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Optional

from loguru import logger
from sqlalchemy import func, text

from config.settings import settings
from data.database import get_session, Trade, DailyStats, EngineState


class SafetyManager:
    """Monitors and enforces trading safety limits."""

    def __init__(self) -> None:
        self.cfg = settings.safety
        self._paused: bool = False
        self._killed: bool = False
        self._safe_mode: bool = False
        self._consecutive_losses: int = 0
        self._daily_pnl: float = 0.0
        self._daily_trade_count: int = 0
        self._last_check_date: Optional[date] = None

    def can_trade(self) -> tuple[bool, str]:
        self._refresh_state_from_db()
        if self._killed:
            return False, "Emergency kill switch activated"
        if self._safe_mode:
            return False, "SAFE_MODE active (manual reset required)"
        if self._paused:
            return False, "Trading is paused"

        self._refresh_daily_stats()

        balance = self._get_starting_balance()
        if balance > 0 and self._daily_pnl < 0:
            if abs(self._daily_pnl) / balance > self.cfg.max_daily_loss_pct:
                if not self._paused:
                    logger.critical(
                        f"Daily loss circuit breaker triggered at {self._daily_pnl:.2f} "
                        f"(limit={self.cfg.max_daily_loss_pct:.2%})"
                    )
                    self.pause_trading()
                    self._mark_daily_circuit_breaker()
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
            "safe_mode": self._safe_mode,
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl": self._daily_pnl,
            "daily_trade_count": self._daily_trade_count,
            "max_daily_loss_pct": self.cfg.max_daily_loss_pct,
            "max_consecutive_losses": self.cfg.max_consecutive_losses,
        }

    def load_state(self) -> None:
        self._refresh_state_from_db()
        self._refresh_daily_stats()

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

    def _refresh_state_from_db(self) -> None:
        session = get_session()
        try:
            for key in ("paused", "killed", "safe_mode"):
                try:
                    state = session.query(EngineState).filter(EngineState.key == key).first()
                    value = bool(state.value.get("value", False)) if state else False
                    current = bool(getattr(self, f"_{key}", False))
                    setattr(self, f"_{key}", bool(current or value))
                except Exception as exc:
                    logger.error("Error loading engine state '{}' : {}", key, exc)
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
        from data.database import get_engine

        engine = get_engine()
        today = date.today()
        try:
            with engine.begin() as conn:
                # Upsert: try insert first, update on conflict
                row = conn.execute(
                    text("SELECT id FROM daily_stats WHERE date = :d"),
                    {"d": today},
                ).fetchone()
                if row is None:
                    conn.execute(
                        text(
                            "INSERT INTO daily_stats "
                            "(date, total_trades, winning_trades, losing_trades, "
                            " total_pnl, max_drawdown, consecutive_losses, circuit_breaker_triggered) "
                            "VALUES (:d, 0, 0, 0, 0.0, 0.0, 0, 0)"
                        ),
                        {"d": today},
                    )
                win_col = "winning_trades" if pnl >= 0 else "losing_trades"
                conn.execute(
                    text(
                        f"UPDATE daily_stats SET "
                        f"total_trades = total_trades + 1, "
                        f"total_pnl = total_pnl + :pnl, "
                        f"{win_col} = {win_col} + 1, "
                        f"consecutive_losses = :cl "
                        f"WHERE date = :d"
                    ),
                    {"pnl": pnl, "cl": self._consecutive_losses, "d": today},
                )
        except Exception as e:
            logger.error(f"Error updating daily stats: {e}")

    def _mark_daily_circuit_breaker(self) -> None:
        from data.database import get_engine

        engine = get_engine()
        today = date.today()
        try:
            with engine.begin() as conn:
                row = conn.execute(
                    text("SELECT id FROM daily_stats WHERE date = :d"),
                    {"d": today},
                ).fetchone()
                if row is None:
                    conn.execute(
                        text(
                            "INSERT INTO daily_stats "
                            "(date, total_trades, winning_trades, losing_trades, "
                            " total_pnl, max_drawdown, consecutive_losses, circuit_breaker_triggered) "
                            "VALUES (:d, :tc, 0, 0, :pnl, 0.0, 0, 1)"
                        ),
                        {"d": today, "tc": self._daily_trade_count, "pnl": self._daily_pnl},
                    )
                else:
                    conn.execute(
                        text("UPDATE daily_stats SET circuit_breaker_triggered = 1 WHERE date = :d"),
                        {"d": today},
                    )
        except Exception as exc:
            logger.error(f"Error marking daily circuit breaker: {exc}")
