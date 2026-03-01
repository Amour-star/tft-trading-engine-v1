"""
Deterministic daily strategy parameter evolution.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Optional

from loguru import logger

from config.settings import settings
from data.database import EngineState, StrategyParameter, Trade, get_session

_STATE_LAST_EVOLUTION_KEY = "strategy_last_evolution_date"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_weights(tft: float, xgb: float, ppo: float) -> tuple[float, float, float]:
    tft = max(0.05, _safe_float(tft, 0.4))
    xgb = max(0.05, _safe_float(xgb, 0.4))
    ppo = max(0.05, _safe_float(ppo, 0.2))
    total = tft + xgb + ppo
    if total <= 0:
        return 0.4, 0.4, 0.2
    return tft / total, xgb / total, ppo / total


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_v = sum(values) / len(values)
    variance = sum((v - mean_v) ** 2 for v in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_ret = sum(returns) / len(returns)
    std_ret = _std(returns)
    if std_ret < 1e-12:
        return 0.0
    return mean_ret / std_ret


def _max_drawdown_from_pnl(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        dd = equity - peak
        if dd < max_dd:
            max_dd = dd
    return abs(max_dd)


class StrategyEvolutionEngine:
    """Loads, applies, and evolves strategy parameters."""

    def __init__(self) -> None:
        self.default_tft_weight = 0.4
        self.default_xgb_weight = 0.4
        self.default_ppo_weight = 0.2

    def load_latest_parameters(self) -> Dict[str, float]:
        session = get_session()
        try:
            latest = (
                session.query(StrategyParameter)
                .order_by(StrategyParameter.id.desc())
                .first()
            )
            if latest:
                return {
                    "tft_weight": _safe_float(latest.tft_weight, self.default_tft_weight),
                    "xgb_weight": _safe_float(latest.xgb_weight, self.default_xgb_weight),
                    "ppo_weight": _safe_float(latest.ppo_weight, self.default_ppo_weight),
                    "confidence_threshold": _safe_float(
                        latest.confidence_threshold,
                        settings.trading.confidence_threshold,
                    ),
                    "risk_per_trade": _safe_float(
                        latest.risk_per_trade,
                        settings.trading.risk_per_trade,
                    ),
                }

            tft_w, xgb_w, ppo_w = _normalize_weights(
                self.default_tft_weight,
                self.default_xgb_weight,
                self.default_ppo_weight,
            )
            inserted = StrategyParameter(
                timestamp=datetime.utcnow().isoformat(),
                tft_weight=tft_w,
                xgb_weight=xgb_w,
                ppo_weight=ppo_w,
                confidence_threshold=settings.trading.confidence_threshold,
                risk_per_trade=settings.trading.risk_per_trade,
            )
            session.add(inserted)
            session.commit()
            return {
                "tft_weight": tft_w,
                "xgb_weight": xgb_w,
                "ppo_weight": ppo_w,
                "confidence_threshold": settings.trading.confidence_threshold,
                "risk_per_trade": settings.trading.risk_per_trade,
            }
        except Exception as exc:
            session.rollback()
            logger.error(f"Load strategy parameters failed: {exc}")
            return {
                "tft_weight": self.default_tft_weight,
                "xgb_weight": self.default_xgb_weight,
                "ppo_weight": self.default_ppo_weight,
                "confidence_threshold": settings.trading.confidence_threshold,
                "risk_per_trade": settings.trading.risk_per_trade,
            }
        finally:
            session.close()

    def apply_to_decision(self, decision_engine, params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        chosen = params or self.load_latest_parameters()
        decision_engine.update_thresholds(chosen)
        return chosen

    def evolve_if_due(self, open_trade_count: int) -> Optional[Dict[str, float]]:
        """
        Run once per UTC day and only when no open trades exist.
        """
        if open_trade_count > 0:
            return None

        today = datetime.utcnow().date().isoformat()
        session = get_session()
        try:
            last_state = session.query(EngineState).filter(EngineState.key == _STATE_LAST_EVOLUTION_KEY).first()
            last_evolved = (
                str((last_state.value or {}).get("value", "")).strip()
                if last_state and last_state.value
                else ""
            )
            if last_evolved == today:
                return None

            latest = (
                session.query(StrategyParameter)
                .order_by(StrategyParameter.id.desc())
                .first()
            )
            if latest:
                current = {
                    "tft_weight": _safe_float(latest.tft_weight, self.default_tft_weight),
                    "xgb_weight": _safe_float(latest.xgb_weight, self.default_xgb_weight),
                    "ppo_weight": _safe_float(latest.ppo_weight, self.default_ppo_weight),
                    "confidence_threshold": _safe_float(
                        latest.confidence_threshold,
                        settings.trading.confidence_threshold,
                    ),
                    "risk_per_trade": _safe_float(latest.risk_per_trade, settings.trading.risk_per_trade),
                }
            else:
                current = self.load_latest_parameters()

            trades = (
                session.query(Trade)
                .filter(Trade.status == "closed", Trade.pnl.isnot(None))
                .order_by(Trade.exit_time.desc(), Trade.id.desc())
                .limit(100)
                .all()
            )
            if len(trades) < 20:
                # Mark as run to enforce once-per-day deterministically even with low sample size.
                if last_state:
                    last_state.value = {"value": today}
                else:
                    session.add(EngineState(key=_STATE_LAST_EVOLUTION_KEY, value={"value": today}))
                session.commit()
                return None

            ordered = list(reversed(trades))
            returns = [_safe_float(t.pnl_pct) for t in ordered]
            sharpe_value = _sharpe(returns)

            tft_weight_sum = 0.0
            tft_win_sum = 0.0
            xgb_weight_sum = 0.0
            xgb_win_sum = 0.0
            pnls = []

            for trade in ordered:
                pnl = _safe_float(trade.pnl)
                pnls.append(pnl)
                tft_score = _safe_float(trade.tft_score)
                xgb_score = _safe_float(trade.xgb_score)
                ppo_score = _safe_float(trade.ppo_score)
                gov_score = _safe_float(trade.gov_adjust)
                score_total = tft_score + xgb_score + ppo_score + gov_score
                if abs(score_total) < 1e-12:
                    score_total = 4.0
                    tft_score = xgb_score = ppo_score = gov_score = 1.0

                tft_w = tft_score / score_total
                xgb_w = xgb_score / score_total
                tft_weight_sum += abs(tft_w)
                xgb_weight_sum += abs(xgb_w)
                if pnl * tft_w > 0:
                    tft_win_sum += abs(tft_w)
                if pnl * xgb_w > 0:
                    xgb_win_sum += abs(xgb_w)

            tft_win_rate = (tft_win_sum / tft_weight_sum) if tft_weight_sum > 1e-12 else 0.0
            xgb_win_rate = (xgb_win_sum / xgb_weight_sum) if xgb_weight_sum > 1e-12 else 0.0

            previous_dd = 0.0
            recent_dd = 0.0
            if len(pnls) >= 80:
                mid = len(pnls) // 2
                previous_dd = _max_drawdown_from_pnl(pnls[:mid])
                recent_dd = _max_drawdown_from_pnl(pnls[mid:])

            evolved = dict(current)

            if sharpe_value < 0.5:
                evolved["risk_per_trade"] = max(0.002, _safe_float(evolved["risk_per_trade"]) * 0.8)

            if xgb_win_rate > tft_win_rate:
                evolved["xgb_weight"] = _safe_float(evolved["xgb_weight"]) + 0.05
                evolved["tft_weight"] = _safe_float(evolved["tft_weight"]) - 0.05

            if recent_dd > previous_dd and recent_dd > 0:
                evolved["confidence_threshold"] = min(
                    0.95, _safe_float(evolved["confidence_threshold"]) + 0.02
                )

            tft_w, xgb_w, ppo_w = _normalize_weights(
                _safe_float(evolved["tft_weight"]),
                _safe_float(evolved["xgb_weight"]),
                _safe_float(evolved["ppo_weight"]),
            )
            evolved["tft_weight"] = tft_w
            evolved["xgb_weight"] = xgb_w
            evolved["ppo_weight"] = ppo_w
            evolved["confidence_threshold"] = max(
                0.4, min(0.95, _safe_float(evolved["confidence_threshold"]))
            )
            evolved["risk_per_trade"] = max(
                0.002, min(0.05, _safe_float(evolved["risk_per_trade"]))
            )

            session.add(
                StrategyParameter(
                    timestamp=datetime.utcnow().isoformat(),
                    tft_weight=evolved["tft_weight"],
                    xgb_weight=evolved["xgb_weight"],
                    ppo_weight=evolved["ppo_weight"],
                    confidence_threshold=evolved["confidence_threshold"],
                    risk_per_trade=evolved["risk_per_trade"],
                )
            )

            if last_state:
                last_state.value = {"value": today}
            else:
                session.add(EngineState(key=_STATE_LAST_EVOLUTION_KEY, value={"value": today}))
            session.commit()
            return evolved
        except Exception as exc:
            session.rollback()
            logger.error(f"Strategy evolution failed: {exc}")
            return None
        finally:
            session.close()
