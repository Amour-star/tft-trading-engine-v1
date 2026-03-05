"""
Prop-firm style deterministic risk controls.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger
from sqlalchemy import func

from config.settings import ACTIVE_SYMBOL, settings
from data.database import EngineState, RiskState, Trade, get_session

_STATE_DAY_START_EQUITY = "prop_day_start_equity"
_STATE_MAX_EQUITY = "prop_max_equity"
_STATE_COOLDOWN_UNTIL = "prop_cooldown_until"
_STATE_SINGLE_SYMBOL_LOSS_STREAK = "single_symbol_loss_streak"
_STATE_REBASELINE_REQUESTED = "prop_rebaseline_requested"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def request_prop_risk_rebaseline(reason: str = "manual_reset") -> bool:
    """
    Ask prop-risk controls to reset baselines/cooldowns on the next risk check.
    """
    session = get_session()
    try:
        state = session.query(EngineState).filter(EngineState.key == _STATE_REBASELINE_REQUESTED).first()
        payload = {
            "value": True,
            "reason": str(reason or "manual_reset"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        if state:
            state.value = payload
        else:
            session.add(EngineState(key=_STATE_REBASELINE_REQUESTED, value=payload))
        session.commit()
        logger.bind(event="PROP_RISK_REBASELINE_REQUESTED", reason=payload["reason"]).warning(
            "PROP_RISK_REBASELINE_REQUESTED"
        )
        return True
    except Exception as exc:
        session.rollback()
        logger.error("Failed to request prop-risk rebaseline: {}", exc)
        return False
    finally:
        session.close()


class PropRiskManager:
    """Pre-trade approvals + post-close risk-state updates."""

    def __init__(self, fetcher, executor) -> None:
        self.fetcher = fetcher
        self.executor = executor
        self.symbol = str(os.getenv("SYMBOL", ACTIVE_SYMBOL)).strip() or ACTIVE_SYMBOL
        self.max_daily_loss_pct = _safe_float(
            os.getenv("MAX_DAILY_LOSS_PCT", str(settings.safety.max_daily_loss_pct)),
            settings.safety.max_daily_loss_pct,
        )
        self.trailing_dd_limit_pct = _safe_float(os.getenv("TRAILING_DD_LIMIT_PCT", "0.08"), 0.08)
        # Institutional paper mode: allow up to full account exposure by default.
        self.max_exposure_pct = _safe_float(os.getenv("MAX_EXPOSURE_PCT", "1.0"), 1.0)
        self.loss_streak_limit = int(os.getenv("PROP_MAX_CONSECUTIVE_LOSSES", "3"))
        self.cooldown_hours = int(os.getenv("PROP_COOLDOWN_HOURS", "6"))

    def can_open_new_trade(self, execution_in_progress: bool = False) -> tuple[bool, str]:
        """
        Run all prop-style risk checks prior to submitting a new order.
        """
        if execution_in_progress:
            return True, "Execution in progress; risk state unchanged"

        now = datetime.utcnow()
        today = now.date().isoformat()

        equity = self.estimate_equity()
        rebaseline_requested = bool(self._get_engine_state(_STATE_REBASELINE_REQUESTED))
        if rebaseline_requested:
            self._set_engine_state(_STATE_DAY_START_EQUITY, {"date": today, "equity": float(equity)})
            self._set_engine_state(_STATE_MAX_EQUITY, float(equity))
            self._set_engine_state(_STATE_COOLDOWN_UNTIL, None)
            self._set_engine_state(_STATE_SINGLE_SYMBOL_LOSS_STREAK, 0)
            self._set_engine_state(_STATE_REBASELINE_REQUESTED, False)
            logger.bind(
                event="PROP_RISK_REBASELINED",
                symbol=self.symbol,
                equity=float(equity),
                date=today,
            ).warning("PROP_RISK_REBASELINED")

        daily_loss = self._compute_daily_loss()
        day_start_equity = self._get_day_start_equity(today, equity)
        daily_loss_pct = (daily_loss / day_start_equity) if day_start_equity > 0 else 0.0

        max_equity = self._get_max_equity(equity)
        drawdown_limit = max_equity * self.trailing_dd_limit_pct
        trailing_breached = equity < (max_equity - drawdown_limit)
        exposure = self._compute_open_exposure()
        single_symbol_loss_streak = self._compute_single_symbol_loss_streak()

        cooldown_until = self._get_cooldown_until()
        if cooldown_until and now < cooldown_until:
            self._persist_risk_state(
                date_key=today,
                trading_enabled=False,
                daily_loss=daily_loss,
                max_equity=max_equity,
                consecutive_losses=single_symbol_loss_streak,
            )
            return False, f"Consecutive loss cooldown active until {cooldown_until.isoformat()} UTC"

        if daily_loss_pct > self.max_daily_loss_pct:
            self._persist_risk_state(
                date_key=today,
                trading_enabled=False,
                daily_loss=daily_loss,
                max_equity=max_equity,
                consecutive_losses=single_symbol_loss_streak,
            )
            return False, f"Daily loss limit exceeded ({daily_loss_pct:.2%})"

        if trailing_breached:
            self._persist_risk_state(
                date_key=today,
                trading_enabled=False,
                daily_loss=daily_loss,
                max_equity=max_equity,
                consecutive_losses=single_symbol_loss_streak,
            )
            return False, "Trailing drawdown limit breached"

        if equity > 0 and exposure > (self.max_exposure_pct * equity):
            self._persist_risk_state(
                date_key=today,
                trading_enabled=False,
                daily_loss=daily_loss,
                max_equity=max_equity,
                consecutive_losses=single_symbol_loss_streak,
            )
            return False, "Max exposure limit exceeded"

        self._persist_risk_state(
            date_key=today,
            trading_enabled=True,
            daily_loss=daily_loss,
            max_equity=max_equity,
            consecutive_losses=single_symbol_loss_streak,
        )
        return True, "Prop-risk checks passed"

    def on_trade_closed_by_id(self, trade_id: str) -> None:
        """Look up a trade by ID and delegate to on_trade_closed."""
        session = get_session()
        try:
            trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
            if trade and trade.pnl is not None:
                pnl = float(trade.pnl)
            else:
                return
        finally:
            session.close()
        self._on_trade_closed_with_pnl(pnl)

    def on_trade_closed(self, trade: Trade) -> None:
        """
        Update loss streak/cooldown and refresh persisted risk state.
        """
        pnl = _safe_float(getattr(trade, "pnl", 0.0))
        self._on_trade_closed_with_pnl(pnl)

    def _on_trade_closed_with_pnl(self, pnl: float) -> None:
        """Core post-close logic using a plain pnl value."""
        now = datetime.utcnow()
        today = now.date().isoformat()
        try:
            single_symbol_loss_streak = self._update_single_symbol_loss_streak(pnl)
            if pnl < 0 and single_symbol_loss_streak >= self.loss_streak_limit:
                cooldown_until = now + timedelta(hours=self.cooldown_hours)
                self._set_engine_state(_STATE_COOLDOWN_UNTIL, cooldown_until.isoformat())
                logger.warning(
                    f"Prop risk cooldown triggered ({single_symbol_loss_streak} losses): "
                    f"until {cooldown_until.isoformat()} UTC"
                )

            equity = self.estimate_equity()
            daily_loss = self._compute_daily_loss()
            max_equity = self._get_max_equity(equity)
            self._persist_risk_state(
                date_key=today,
                trading_enabled=True,
                daily_loss=daily_loss,
                max_equity=max_equity,
                consecutive_losses=single_symbol_loss_streak,
            )
        except Exception as exc:
            logger.error(f"Prop risk post-close update failed: {exc}")

    def estimate_equity(self) -> float:
        """
        Estimate account equity as cash balance + marked value of open positions.
        """
        balance = _safe_float(self.executor.get_balance(), settings.trading.paper_starting_balance)
        marked_value = 0.0
        for symbol, qty, fallback_entry in self._iter_open_positions():
            try:
                ticker = self.fetcher.get_ticker(symbol)
                price = _safe_float(ticker.get("price"), fallback_entry)
            except Exception:
                price = fallback_entry
            marked_value += price * qty
        return max(0.0, balance + marked_value)

    def _compute_daily_loss(self) -> float:
        session = get_session()
        try:
            today = datetime.utcnow().date()
            today_start = datetime.combine(today, datetime.min.time())
            pnl_sum = (
                session.query(func.sum(Trade.pnl))
                .filter(
                    Trade.status == "closed",
                    Trade.pair == self.symbol,
                    Trade.exit_time >= today_start,
                )
                .scalar()
            )
            pnl = _safe_float(pnl_sum, 0.0)
            return abs(pnl) if pnl < 0 else 0.0
        finally:
            session.close()

    def _compute_open_exposure(self) -> float:
        exposure = 0.0
        for symbol, qty, fallback_entry in self._iter_open_positions():
            try:
                ticker = self.fetcher.get_ticker(symbol)
                mark = _safe_float(ticker.get("price"), fallback_entry)
            except Exception:
                mark = fallback_entry
            exposure += abs(mark * qty)
        return max(0.0, exposure)

    def _iter_open_positions(self) -> list[tuple[str, float, float]]:
        """
        Return open positions as tuples of (symbol, signed_qty, fallback_entry_price).
        PAPER mode uses the paper snapshot as source of truth to avoid stale Trade rows.
        """
        if str(settings.trading.trading_mode).upper() == "PAPER":
            try:
                from execution.paper_executor import load_paper_snapshot

                snapshot = load_paper_snapshot()
                rows: list[tuple[str, float, float]] = []
                for symbol, payload in (snapshot.get("positions", {}) or {}).items():
                    qty = _safe_float((payload or {}).get("quantity"), 0.0)
                    if abs(qty) <= 1e-12:
                        continue
                    entry = _safe_float((payload or {}).get("avg_entry_price"), 0.0)
                    rows.append((str(symbol or self.symbol), qty, entry))
                return rows
            except Exception as exc:
                logger.debug("Falling back to Trade rows for prop-risk position view: {}", exc)

        session = get_session()
        try:
            open_trades = (
                session.query(Trade)
                .filter(Trade.status == "open", Trade.pair == self.symbol)
                .all()
            )
            rows = []
            for trade in open_trades:
                qty = _safe_float(trade.quantity, 0.0)
                side = str(getattr(trade, "side", "BUY")).upper()
                signed_qty = qty if side != "SELL" else -qty
                if abs(signed_qty) <= 1e-12:
                    continue
                rows.append((str(trade.pair or self.symbol), signed_qty, _safe_float(trade.entry_price, 0.0)))
            return rows
        finally:
            session.close()

    def _compute_single_symbol_loss_streak(self) -> int:
        persisted = self._get_engine_state(_STATE_SINGLE_SYMBOL_LOSS_STREAK)
        if persisted is None:
            rebuilt = self._rebuild_single_symbol_loss_streak_from_trades()
            self._set_engine_state(_STATE_SINGLE_SYMBOL_LOSS_STREAK, int(rebuilt))
            return int(rebuilt)
        return max(0, int(_safe_float(persisted, 0.0)))

    def _update_single_symbol_loss_streak(self, pnl: float) -> int:
        streak = self._compute_single_symbol_loss_streak()
        if pnl < 0:
            streak += 1
        else:
            streak = 0
        self._set_engine_state(_STATE_SINGLE_SYMBOL_LOSS_STREAK, int(streak))
        return streak

    def _compute_consecutive_losses(self) -> int:
        """Backward-compatible alias used by old callers/tests."""
        return self._compute_single_symbol_loss_streak()

    def _rebuild_single_symbol_loss_streak_from_trades(self) -> int:
        session = get_session()
        try:
            recent = (
                session.query(Trade)
                .filter(
                    Trade.status == "closed",
                    Trade.pair == self.symbol,
                    Trade.pnl.isnot(None),
                )
                .order_by(Trade.exit_time.desc(), Trade.id.desc())
                .limit(max(20, self.loss_streak_limit + 5))
                .all()
            )
            streak = 0
            for trade in recent:
                if _safe_float(trade.pnl) < 0:
                    streak += 1
                else:
                    break
            return streak
        finally:
            session.close()

    def _get_day_start_equity(self, today: str, default_equity: float) -> float:
        state = self._get_engine_state(_STATE_DAY_START_EQUITY)
        if isinstance(state, dict) and str(state.get("date")) == today:
            return _safe_float(state.get("equity"), default_equity)

        self._set_engine_state(_STATE_DAY_START_EQUITY, {"date": today, "equity": default_equity})
        return default_equity

    def _get_max_equity(self, current_equity: float) -> float:
        prev = _safe_float(self._get_engine_state(_STATE_MAX_EQUITY), current_equity)
        max_equity = max(prev, current_equity)
        self._set_engine_state(_STATE_MAX_EQUITY, max_equity)
        return max_equity

    def _get_cooldown_until(self) -> Optional[datetime]:
        raw = self._get_engine_state(_STATE_COOLDOWN_UNTIL)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return None

    def _persist_risk_state(
        self,
        date_key: str,
        trading_enabled: bool,
        daily_loss: float,
        max_equity: float,
        consecutive_losses: int,
    ) -> None:
        session = get_session()
        try:
            row = session.query(RiskState).filter(RiskState.date == date_key).first()
            if row:
                row.trading_enabled = bool(trading_enabled)
                row.daily_loss = float(daily_loss)
                row.max_equity = float(max_equity)
                row.consecutive_losses = int(consecutive_losses)
            else:
                session.add(
                    RiskState(
                        date=date_key,
                        trading_enabled=bool(trading_enabled),
                        daily_loss=float(daily_loss),
                        max_equity=float(max_equity),
                        consecutive_losses=int(consecutive_losses),
                    )
                )
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Persist risk_state failed: {exc}")
        finally:
            session.close()

    def _get_engine_state(self, key: str):
        session = get_session()
        try:
            state = session.query(EngineState).filter(EngineState.key == key).first()
            return (state.value or {}).get("value") if state and state.value else None
        finally:
            session.close()

    def _set_engine_state(self, key: str, value) -> None:
        session = get_session()
        try:
            state = session.query(EngineState).filter(EngineState.key == key).first()
            if state:
                state.value = {"value": value}
            else:
                session.add(EngineState(key=key, value={"value": value}))
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Persist EngineState {key} failed: {exc}")
        finally:
            session.close()
