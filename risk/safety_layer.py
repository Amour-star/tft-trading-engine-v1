"""
Safety layer module: deterministic pre-trade guards + global kill switch.

All execution paths (paper/live) should call these checks before placing orders.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from loguru import logger
from sqlalchemy import func

from data.database import EngineState, Trade, get_session


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class SafetyLimits:
    max_position_pct: float
    max_notional_per_trade: float
    kill_switch_daily_drawdown: float
    kill_switch_balance_drop: float
    max_price: float = 1_000_000.0
    max_jump_pct: float = 0.50


def load_limits() -> SafetyLimits:
    return SafetyLimits(
        # Default to full-balance sizing for paper execution unless explicitly overridden.
        max_position_pct=max(0.0, _env_float("MAX_POSITION_PCT", 1.0)),
        max_notional_per_trade=max(0.0, _env_float("MAX_NOTIONAL_PER_TRADE", 1_000_000_000.0)),
        kill_switch_daily_drawdown=max(0.0, _env_float("KILL_SWITCH_DAILY_DRAWDOWN", 0.15)),
        kill_switch_balance_drop=max(0.0, _env_float("KILL_SWITCH_BALANCE_DROP", 0.30)),
    )


def validate_price(pair: str, price: float, last_price: Optional[float]) -> bool:
    """
    Sanity-check trade prices.

    Rules:
    - Reject price <= 0
    - Reject price > 1_000_000
    - Reject if price jump > 50% from last known price
    """
    limits = load_limits()
    value = float(price or 0.0)
    if value <= 0.0 or value > limits.max_price:
        logger.bind(
            event="PRICE_REJECTED",
            pair=pair,
            price=value,
            last_price=last_price,
            reason="out_of_bounds",
        ).error("PRICE_REJECTED")
        return False

    if last_price is not None:
        prev = float(last_price or 0.0)
        if prev > 0:
            jump_pct = abs(value - prev) / prev
            if jump_pct > limits.max_jump_pct:
                logger.bind(
                    event="PRICE_REJECTED",
                    pair=pair,
                    price=value,
                    last_price=prev,
                    jump_pct=jump_pct,
                    reason="jump_exceeds_limit",
                ).error("PRICE_REJECTED")
                return False

    return True


def validate_position_size(pair: str, notional: float, balance: float) -> bool:
    """
    Enforce environment-driven trade caps.

    Blocks if:
    - notional > balance * MAX_POSITION_PCT
    - notional > MAX_NOTIONAL_PER_TRADE
    """
    limits = load_limits()
    notional_value = float(notional or 0.0)
    balance_value = float(balance or 0.0)
    if notional_value <= 0.0:
        logger.bind(
            event="TRADE_BLOCKED",
            pair=pair,
            notional=notional_value,
            balance=balance_value,
            reason="non_positive_notional",
        ).warning("TRADE_BLOCKED")
        return False

    max_by_pct = balance_value * float(limits.max_position_pct)
    if max_by_pct > 0 and notional_value > max_by_pct + 1e-12:
        logger.bind(
            event="TRADE_BLOCKED",
            pair=pair,
            notional=notional_value,
            balance=balance_value,
            max_position_pct=limits.max_position_pct,
            max_notional=max_by_pct,
            reason="max_position_pct",
        ).warning("TRADE_BLOCKED")
        return False

    if limits.max_notional_per_trade > 0 and notional_value > limits.max_notional_per_trade + 1e-12:
        logger.bind(
            event="TRADE_BLOCKED",
            pair=pair,
            notional=notional_value,
            balance=balance_value,
            max_notional_per_trade=limits.max_notional_per_trade,
            reason="max_notional_per_trade",
        ).warning("TRADE_BLOCKED")
        return False

    return True


def abnormal_trade_detector(
    pair: str,
    price: float,
    qty: float,
    balance: float,
) -> bool:
    """
    Lightweight anomaly detector (unit-testable heuristic).

    Returns True if trade looks abnormal and should be blocked.
    """
    price_value = float(price or 0.0)
    qty_value = float(qty or 0.0)
    balance_value = float(balance or 0.0)

    if qty_value <= 0 or price_value <= 0:
        return True

    notional = price_value * qty_value
    near_full_pct = max(0.50, min(1.50, _env_float("ABNORMAL_NOTIONAL_PCT", 0.999)))
    if balance_value > 0 and notional > balance_value * near_full_pct:
        logger.bind(
            event="TRADE_BLOCKED",
            pair=pair,
            price=price_value,
            qty=qty_value,
            balance=balance_value,
            notional=notional,
            near_full_pct=near_full_pct,
            reason="near_total_balance",
        ).warning("TRADE_BLOCKED")
        return True

    return False


def _get_state_value(session, key: str) -> Optional[Dict[str, Any]]:
    row = session.query(EngineState).filter(EngineState.key == key).first()
    if not row:
        return None
    try:
        value = row.value
    except Exception:
        value = None
    if isinstance(value, dict):
        return value
    return None


def _upsert_state_value(session, key: str, value: Dict[str, Any]) -> None:
    row = session.query(EngineState).filter(EngineState.key == key).first()
    if row:
        row.value = value
        row.updated_at = datetime.utcnow()
    else:
        session.add(EngineState(key=key, value=value))


def _daily_loss_utc(session) -> float:
    today = datetime.utcnow().date()
    today_start = datetime.combine(today, datetime.min.time())
    pnl_sum = (
        session.query(func.sum(Trade.pnl))
        .filter(Trade.status == "closed", Trade.exit_time >= today_start)
        .scalar()
    )
    pnl = float(pnl_sum or 0.0)
    return abs(pnl) if pnl < 0 else 0.0


def kill_switch_check(
    equity: float,
    day_start_equity: float,
    initial_equity: float,
) -> Tuple[bool, str]:
    """
    Pure check (no DB): evaluate kill-switch thresholds.
    """
    limits = load_limits()
    equity_value = float(equity or 0.0)
    day0 = float(day_start_equity or 0.0)
    init0 = float(initial_equity or 0.0)

    if day0 > 0 and limits.kill_switch_daily_drawdown > 0:
        if equity_value < day0 * (1.0 - limits.kill_switch_daily_drawdown):
            return True, "daily_drawdown"

    if init0 > 0 and limits.kill_switch_balance_drop > 0:
        if equity_value < init0 * (1.0 - limits.kill_switch_balance_drop):
            return True, "balance_drop"

    return False, ""


def evaluate_and_arm_kill_switch(
    equity: float,
    *,
    now: Optional[datetime] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    DB-backed kill switch evaluation.

    On trigger:
    - sets engine_state.safe_mode = True
    - sets engine_state.paused = True

    Returns (triggered, reason, context).
    """
    limits = load_limits()
    now_dt = now or datetime.now(timezone.utc)
    today = now_dt.date().isoformat()
    equity_value = float(equity or 0.0)

    session = get_session()
    try:
        rebaseline_requested = False
        rebaseline_reason = ""
        day_start = 0.0
        initial_equity = 0.0

        rebaseline_state = _get_state_value(session, "kill_switch_rebaseline")
        if isinstance(rebaseline_state, dict) and bool(rebaseline_state.get("value")):
            rebaseline_requested = True
            rebaseline_reason = str(rebaseline_state.get("reason") or "manual_reset")
            day_start = equity_value
            initial_equity = equity_value
            _upsert_state_value(
                session,
                "kill_switch_day_start_equity",
                {"date": today, "equity": equity_value},
            )
            _upsert_state_value(session, "kill_switch_initial_equity", {"equity": equity_value})
            _upsert_state_value(
                session,
                "kill_switch_rebaseline",
                {
                    "value": False,
                    "reason": rebaseline_reason,
                    "applied_at": now_dt.isoformat(),
                    "equity": equity_value,
                },
            )
            logger.bind(
                event="KILL_SWITCH_REBASELINED",
                reason=rebaseline_reason,
                equity=equity_value,
                date=today,
            ).warning("KILL_SWITCH_REBASELINED")
        else:
            # Day-start baseline (UTC).
            day_state = _get_state_value(session, "kill_switch_day_start_equity")
            if isinstance(day_state, dict) and str(day_state.get("date")) == today:
                day_start = float(day_state.get("equity") or 0.0)
            else:
                day_start = equity_value
                _upsert_state_value(
                    session,
                    "kill_switch_day_start_equity",
                    {"date": today, "equity": equity_value},
                )

            # Initial baseline (first-seen equity).
            initial_state = _get_state_value(session, "kill_switch_initial_equity")
            if isinstance(initial_state, dict) and initial_state.get("equity") is not None:
                initial_equity = float(initial_state.get("equity") or 0.0)
            else:
                initial_equity = equity_value
                _upsert_state_value(session, "kill_switch_initial_equity", {"equity": equity_value})

        triggered = False
        reason = ""
        triggered, reason = kill_switch_check(equity_value, day_start, initial_equity)

        context = {
            "equity": equity_value,
            "day_start_equity": day_start,
            "initial_equity": initial_equity,
            "daily_drawdown": ((day_start - equity_value) / day_start) if day_start > 0 else 0.0,
            "rebaseline_applied": rebaseline_requested,
            "rebaseline_reason": rebaseline_reason if rebaseline_requested else None,
            "limits": {
                "kill_switch_daily_drawdown": limits.kill_switch_daily_drawdown,
                "kill_switch_balance_drop": limits.kill_switch_balance_drop,
            },
        }

        if not triggered:
            session.commit()
            return False, "", context

        # Arm SAFE_MODE (idempotent).
        safe_state = _get_state_value(session, "safe_mode")
        already_safe = bool(safe_state and safe_state.get("value") is True)
        if not already_safe:
            _upsert_state_value(
                session,
                "safe_mode",
                {
                    "value": True,
                    "reason": reason,
                    "context": context,
                    "timestamp": now_dt.isoformat(),
                },
            )
            _upsert_state_value(session, "paused", {"value": True})

        session.commit()
        logger.bind(event="SAFE_MODE_ARMED", reason=reason, **context).critical("SAFE_MODE_ARMED")
        return True, reason, context
    except Exception:
        session.rollback()
        logger.exception("Kill-switch evaluation failed")
        return False, "", {"equity": equity_value}
    finally:
        session.close()


def request_kill_switch_rebaseline(reason: str = "manual_reset") -> bool:
    """
    Ask the safety layer to re-baseline kill-switch equity references on next cycle.
    """
    session = get_session()
    now_dt = datetime.now(timezone.utc)
    reason_text = str(reason or "manual_reset")
    try:
        _upsert_state_value(
            session,
            "kill_switch_rebaseline",
            {"value": True, "reason": reason_text, "timestamp": now_dt.isoformat()},
        )
        session.commit()
        logger.bind(
            event="KILL_SWITCH_REBASELINE_REQUESTED",
            reason=reason_text,
            timestamp=now_dt.isoformat(),
        ).warning("KILL_SWITCH_REBASELINE_REQUESTED")
        return True
    except Exception:
        session.rollback()
        logger.exception("Failed to request kill-switch rebaseline")
        return False
    finally:
        session.close()


def is_safe_mode() -> bool:
    session = get_session()
    try:
        row = session.query(EngineState).filter(EngineState.key == "safe_mode").first()
        return bool(row and isinstance(row.value, dict) and row.value.get("value") is True)
    finally:
        session.close()
