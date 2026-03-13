"""
Self-improving feedback loop.
Analyzes trade outcomes, adjusts thresholds, and triggers retraining.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
from loguru import logger

from config.settings import settings
from data.database import LearningMetric, ModelMetric, Trade, get_session


class FeedbackLoop:
    """
    Post-trade analysis and self-improvement engine.
    Compares forecasts vs outcomes, identifies systematic errors,
    and adjusts decision engine thresholds dynamically.
    """

    def __init__(self) -> None:
        self.analysis_window: int = 20
        self.adjustment_rate: float = 0.05

    def analyze_trade(self, trade_id: str) -> Dict[str, Any]:
        """
        Analyze a single completed trade.
        Returns analysis with recommended adjustments.
        """
        session = get_session()
        try:
            trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
            if not trade or trade.status != "closed":
                return {}

            analysis: Dict[str, Any] = {
                "trade_id": trade_id,
                "pair": trade.pair,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "r_multiple": trade.r_multiple,
                "exit_reason": trade.exit_reason,
            }

            prediction = trade.prediction or {}
            features = trade.features_at_entry or {}

            vol_regime = features.get("volatility_regime", "unknown")
            exit_reason = trade.exit_reason or ""
            volatility_misread = (
                vol_regime in ("low", "normal")
                and exit_reason == "stop"
                and trade.pnl is not None
                and trade.pnl < 0
            )
            analysis["volatility_misread"] = volatility_misread

            confidence = trade.confidence or 0
            was_loss = trade.pnl is not None and trade.pnl < 0
            confidence_overestimated = confidence > 0.8 and was_loss
            analysis["confidence_overestimated"] = confidence_overestimated

            if trade.entry_price and trade.stop_price and trade.exit_price:
                if exit_reason == "stop" and was_loss:
                    analysis["stop_too_tight"] = trade.r_multiple is not None and trade.r_multiple > -0.5
                else:
                    analysis["stop_too_tight"] = False
            else:
                analysis["stop_too_tight"] = False

            analysis["stop_too_wide"] = (
                exit_reason == "stop"
                and trade.r_multiple is not None
                and trade.r_multiple < -1.5
            )

            expected_move = prediction.get("expected_move", 0)
            if trade.entry_price and trade.exit_price:
                actual_move = (trade.exit_price - trade.entry_price) / trade.entry_price
                direction_correct = (expected_move > 0 and actual_move > 0) or (
                    expected_move < 0 and actual_move < 0
                )
                analysis["direction_correct"] = direction_correct
                analysis["expected_move"] = expected_move
                analysis["actual_move"] = actual_move
                analysis["forecast_error"] = abs(expected_move - actual_move)
            else:
                analysis["direction_correct"] = None

            metric = LearningMetric(
                trade_id=trade_id,
                forecast_accuracy=1.0 if analysis.get("direction_correct") else 0.0,
                volatility_misread=volatility_misread,
                confidence_overestimated=confidence_overestimated,
                stop_too_tight=analysis.get("stop_too_tight", False),
                stop_too_wide=analysis.get("stop_too_wide", False),
                recommended_adjustments=self._compute_adjustments([analysis]),
                analysis_notes=str(analysis),
            )
            session.add(metric)
            session.commit()
            return analysis
        except Exception as exc:
            session.rollback()
            logger.error(f"Error analyzing trade {trade_id}: {exc}")
            return {}
        finally:
            session.close()

    def compute_trade_statistics(self, limit: int = 100) -> Dict[str, float]:
        """Aggregate trade statistics for reporting and adaptive thresholds."""
        session = get_session()
        try:
            closed_trades = (
                session.query(Trade)
                .filter(Trade.status == "closed", Trade.pnl.isnot(None))
                .order_by(Trade.exit_time.asc(), Trade.id.asc())
                .limit(max(1, int(limit)))
                .all()
            )
        finally:
            session.close()

        if not closed_trades:
            return {
                "sample_size": 0.0,
                "win_rate": 0.0,
                "avg_r": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "average_pnl": 0.0,
            }

        pnls = [float(t.pnl or 0.0) for t in closed_trades]
        wins = [p for p in pnls if p > 0.0]
        losses = [p for p in pnls if p < 0.0]
        r_values = [float(t.r_multiple or 0.0) for t in closed_trades if t.r_multiple is not None]

        equity = max(float(settings.trading.paper_starting_balance or 1.0), 1.0)
        peak = equity
        max_drawdown = 0.0
        returns: List[float] = []
        for pnl in pnls:
            returns.append(pnl / equity if abs(equity) > 1e-9 else 0.0)
            equity += pnl
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = min(max_drawdown, (equity - peak) / peak)

        mean_return = float(np.mean(returns)) if returns else 0.0
        std_return = float(np.std(returns)) if len(returns) > 1 else 0.0
        sharpe_ratio = 0.0
        if std_return > 1e-9:
            sharpe_ratio = float((mean_return / std_return) * math.sqrt(len(returns)))

        gross_profit = float(sum(wins))
        gross_loss = abs(float(sum(losses)))
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else (999.0 if gross_profit > 0 else 0.0)

        return {
            "sample_size": float(len(closed_trades)),
            "win_rate": float(len(wins) / len(closed_trades)),
            "avg_r": float(np.mean(r_values)) if r_values else 0.0,
            "profit_factor": float(profit_factor),
            "max_drawdown": abs(float(max_drawdown)),
            "sharpe_ratio": float(sharpe_ratio) if math.isfinite(sharpe_ratio) else 0.0,
            "average_pnl": float(np.mean(pnls)) if pnls else 0.0,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }

    def compute_batch_adjustments(self) -> Dict[str, float]:
        """
        Analyze recent trades and compute threshold adjustments.
        Called periodically (e.g., every 10 trades or daily).
        """
        session = get_session()
        try:
            recent_trades = (
                session.query(Trade)
                .filter(Trade.status == "closed")
                .order_by(Trade.exit_time.desc())
                .limit(self.analysis_window)
                .all()
            )

            if len(recent_trades) < 5:
                logger.info("Not enough trades for batch analysis")
                return {}

            stats = self.compute_trade_statistics(limit=self.analysis_window)
            win_rate = float(stats.get("win_rate", 0.0))
            avg_r = float(stats.get("avg_r", 0.0))
            profit_factor = float(stats.get("profit_factor", 0.0))
            drawdown = float(stats.get("max_drawdown", 0.0))
            sharpe_ratio = float(stats.get("sharpe_ratio", 0.0))
            avg_confidence = float(np.mean([t.confidence or 0 for t in recent_trades]))

            losing_trades = [t for t in recent_trades if t.pnl and t.pnl < 0]
            stop_exits = sum(1 for t in losing_trades if t.exit_reason == "stop")

            adjustments: Dict[str, float] = {}
            current_conf = float(settings.trading.confidence_threshold)
            current_risk = float(settings.trading.risk_per_trade)
            current_spread = float(settings.trading.max_spread_pct)

            if profit_factor < 1.0 or win_rate < 0.45 or sharpe_ratio < 0.5:
                adjustments["confidence_threshold"] = min(
                    0.85,
                    max(current_conf, avg_confidence) + self.adjustment_rate / 2,
                )
                adjustments["min_confidence"] = min(0.82, current_conf + self.adjustment_rate / 2)
            elif profit_factor > 1.35 and win_rate > 0.55 and sharpe_ratio > 1.0:
                adjustments["confidence_threshold"] = max(0.48, current_conf - self.adjustment_rate / 2)
                adjustments["min_confidence"] = max(0.45, current_conf - self.adjustment_rate / 3)

            if stop_exits > len(losing_trades) * 0.8 and len(losing_trades) > 3:
                adjustments["widen_stops"] = True
                logger.info("Recommending wider stops: high stop-out rate")

            if drawdown > 0.08 or profit_factor < 0.95:
                adjustments["risk_per_trade"] = max(0.005, current_risk * 0.85)
            elif win_rate > 0.55 and avg_r > 1.1 and sharpe_ratio > 1.0:
                adjustments["risk_per_trade"] = min(0.02, current_risk * 1.05)

            if profit_factor < 1.0:
                adjustments["spread_max_pct"] = max(0.0008, current_spread * 0.9)
            elif profit_factor > 1.4:
                adjustments["spread_max_pct"] = min(0.0035, current_spread * 1.05)

            if win_rate < 0.45:
                adjustments["min_volume_ratio"] = 0.85
            elif profit_factor > 1.35 and sharpe_ratio > 1.0:
                adjustments["min_volume_ratio"] = 0.70

            logger.info(
                f"Batch analysis: {len(recent_trades)} trades, "
                f"win_rate={win_rate:.2f}, avg_R={avg_r:.2f}, "
                f"profit_factor={profit_factor:.2f}, sharpe={sharpe_ratio:.2f}, "
                f"drawdown={drawdown:.2%}, adjustments={adjustments}"
            )
            return adjustments
        except Exception as exc:
            logger.error(f"Error in batch analysis: {exc}")
            return {}
        finally:
            session.close()

    def _compute_adjustments(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute adjustments from a list of trade analyses."""
        adj: Dict[str, Any] = {}
        vol_misreads = sum(1 for a in analyses if a.get("volatility_misread"))
        conf_overest = sum(1 for a in analyses if a.get("confidence_overestimated"))
        stops_tight = sum(1 for a in analyses if a.get("stop_too_tight"))

        if vol_misreads > len(analyses) * 0.3:
            adj["increase_volatility_filter"] = True
        if conf_overest > len(analyses) * 0.3:
            adj["raise_confidence_threshold"] = True
        if stops_tight > len(analyses) * 0.3:
            adj["widen_stops"] = True
        return adj

    def should_retrain(self) -> bool:
        """Check if model retraining is needed based on schedule and performance."""
        session = get_session()
        try:
            latest_model = (
                session.query(ModelMetric)
                .filter(ModelMetric.is_active == True)
                .order_by(ModelMetric.trained_at.desc())
                .first()
            )
            if not latest_model:
                return True

            days_since = (datetime.utcnow() - latest_model.trained_at).days
            if days_since >= settings.model.retrain_interval_days:
                logger.info(f"Retraining due: {days_since} days since last training")
                return True

            recent_trades = (
                session.query(Trade)
                .filter(
                    Trade.status == "closed",
                    Trade.exit_time >= datetime.utcnow() - timedelta(days=7),
                )
                .all()
            )
            if len(recent_trades) >= 10:
                stats = self.compute_trade_statistics(limit=len(recent_trades))
                if stats.get("win_rate", 0.0) < 0.40 or stats.get("profit_factor", 0.0) < 1.0:
                    logger.info(
                        "Retraining due: degraded performance win_rate={:.2f} profit_factor={:.2f}",
                        stats.get("win_rate", 0.0),
                        stats.get("profit_factor", 0.0),
                    )
                    return True

            return False
        finally:
            session.close()
