"""
Self-improving feedback loop.
Analyzes trade outcomes, adjusts thresholds, and triggers retraining.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from config.settings import settings
from data.database import get_session, Trade, LearningMetric, ModelMetric


class FeedbackLoop:
    """
    Post-trade analysis and self-improvement engine.
    Compares forecasts vs outcomes, identifies systematic errors,
    and adjusts decision engine thresholds dynamically.
    """

    def __init__(self) -> None:
        self.analysis_window: int = 20  # Number of recent trades to analyze
        self.adjustment_rate: float = 0.05  # Max adjustment per cycle

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

            # 1. Was volatility misread?
            vol_regime = features.get("volatility_regime", "unknown")
            exit_reason = trade.exit_reason or ""
            volatility_misread = (
                vol_regime in ("low", "normal") and exit_reason == "stop"
                and trade.pnl is not None and trade.pnl < 0
            )
            analysis["volatility_misread"] = volatility_misread

            # 2. Was confidence overestimated?
            confidence = trade.confidence or 0
            was_loss = trade.pnl is not None and trade.pnl < 0
            confidence_overestimated = confidence > 0.8 and was_loss
            analysis["confidence_overestimated"] = confidence_overestimated

            # 3. Was stop too tight?
            if trade.entry_price and trade.stop_price and trade.exit_price:
                stop_distance = abs(trade.entry_price - trade.stop_price)
                if exit_reason == "stop" and was_loss:
                    # Check if price eventually moved in our direction
                    # (Would need post-exit data, approximate with R multiple)
                    stop_too_tight = trade.r_multiple is not None and trade.r_multiple > -0.5
                    analysis["stop_too_tight"] = stop_too_tight
                else:
                    analysis["stop_too_tight"] = False
            else:
                analysis["stop_too_tight"] = False

            # 4. Was stop too wide?
            analysis["stop_too_wide"] = (
                exit_reason == "stop" and trade.r_multiple is not None and trade.r_multiple < -1.5
            )

            # 5. Forecast accuracy
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

            # Save learning metric
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

        except Exception as e:
            session.rollback()
            logger.error(f"Error analyzing trade {trade_id}: {e}")
            return {}
        finally:
            session.close()

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

            # Gather metrics
            wins = sum(1 for t in recent_trades if t.pnl and t.pnl > 0)
            losses = len(recent_trades) - wins
            win_rate = wins / len(recent_trades)

            avg_confidence = np.mean([t.confidence or 0 for t in recent_trades])
            avg_r = np.mean([t.r_multiple or 0 for t in recent_trades if t.r_multiple is not None])

            # Analyze losing trades
            losing_trades = [t for t in recent_trades if t.pnl and t.pnl < 0]
            stop_exits = sum(1 for t in losing_trades if t.exit_reason == "stop")

            adjustments: Dict[str, float] = {}

            # Confidence threshold adjustment
            if win_rate < 0.4 and avg_confidence > 0.7:
                # Losing despite high confidence → raise threshold
                adjustments["confidence_threshold"] = min(0.90, avg_confidence + self.adjustment_rate)
                logger.info(f"Raising confidence threshold: win_rate={win_rate:.2f}")

            elif win_rate > 0.65 and avg_confidence > 0.75:
                # Winning with high confidence → can lower slightly
                adjustments["confidence_threshold"] = max(0.60, avg_confidence - self.adjustment_rate / 2)

            # Stop loss adjustment
            if stop_exits > len(losing_trades) * 0.8 and len(losing_trades) > 3:
                # Most losses are stops → stops might be too tight
                adjustments["widen_stops"] = True
                logger.info("Recommending wider stops: high stop-out rate")

            # Risk per trade adjustment
            if win_rate < 0.35:
                adjustments["risk_per_trade"] = max(0.005, settings.trading.risk_per_trade * 0.8)
                logger.info("Reducing risk per trade due to low win rate")
            elif win_rate > 0.6 and avg_r > 1.5:
                adjustments["risk_per_trade"] = min(0.02, settings.trading.risk_per_trade * 1.1)

            logger.info(
                f"Batch analysis: {len(recent_trades)} trades, "
                f"win_rate={win_rate:.2f}, avg_R={avg_r:.2f}, "
                f"adjustments={adjustments}"
            )

            return adjustments

        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
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

            # Check degraded performance
            recent_trades = (
                session.query(Trade)
                .filter(Trade.status == "closed", Trade.exit_time >= datetime.utcnow() - timedelta(days=7))
                .all()
            )
            if len(recent_trades) >= 10:
                win_rate = sum(1 for t in recent_trades if t.pnl and t.pnl > 0) / len(recent_trades)
                if win_rate < 0.35:
                    logger.info(f"Retraining due: poor win rate {win_rate:.2f}")
                    return True

            return False
        finally:
            session.close()
