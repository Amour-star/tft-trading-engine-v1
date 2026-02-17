"""
AI Decision Engine.
Evaluates TFT predictions, selects optimal pairs, and generates trade signals.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from data.fetcher import KuCoinDataFetcher
from data.features import compute_features
from models.tft_model import TFTPredictor


@dataclass
class TradeSignal:
    """A fully evaluated trade signal."""
    pair: str
    side: str  # "BUY" or "SELL" (spot = BUY only for now)
    entry_price: float
    stop_price: float
    target_price: float
    confidence: float
    expected_move: float
    prob_up: float
    prob_down: float
    risk_reward: float
    atr: float
    volatility_regime: str
    market_regime: str
    btc_correlation: float
    spread_pct: float
    volume_24h: float
    reasoning: str
    features_snapshot: Dict[str, Any] = field(default_factory=dict)
    forecast_vector: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PairEvaluation:
    """Evaluation result for a single pair."""
    pair: str
    score: float
    prediction: Dict[str, Any]
    features: Dict[str, Any]
    reasons: List[str]
    disqualified: bool = False
    disqualify_reason: str = ""


class DecisionEngine:
    """
    Core AI decision engine.
    Evaluates market conditions, selects pairs, and generates trade signals.
    """

    def __init__(
        self,
        fetcher: KuCoinDataFetcher,
        predictor: TFTPredictor,
    ) -> None:
        self.fetcher = fetcher
        self.predictor = predictor
        self.confidence_threshold = settings.trading.confidence_threshold
        self.risk_per_trade = settings.trading.risk_per_trade

        # Dynamic thresholds (adjusted by self-improving loop)
        self._atr_min_multiplier: float = 1.0
        self._spread_max_pct: float = 0.005  # 0.5% (was 0.3% – too tight for altcoins)
        self._min_volume_ratio: float = 0.5
        self._btc_corr_weight: float = 0.15

    # ------------------------------------------------------------------
    # Main signal generation pipeline
    # ------------------------------------------------------------------

    def generate_signal(self) -> Optional[TradeSignal]:
        """
        Full pipeline: scan pairs → evaluate → select best → generate signal.
        Returns None if no valid signal found.
        """
        logger.info("Starting signal generation cycle")

        # 1. Get candidate pairs
        candidates = self._get_candidate_pairs()
        if not candidates:
            logger.info("No candidate pairs found")
            return None

        # Pre-fetch BTC data once for the whole cycle (avoids 30 redundant API calls)
        try:
            btc_df = self.fetcher.fetch_klines(
                "BTC-USDT", "15min",
                start_dt=datetime.utcnow() - timedelta(hours=60),
            )
        except Exception as e:
            logger.warning(f"Could not fetch BTC data: {e}")
            btc_df = pd.DataFrame()

        # 2. Evaluate each pair
        evaluations: List[PairEvaluation] = []
        for pair_info in candidates:
            pair = pair_info["symbol"]
            try:
                evaluation = self._evaluate_pair(pair, pair_info, btc_df=btc_df)
                evaluations.append(evaluation)
            except Exception as e:
                logger.warning(f"Error evaluating {pair}: {e}")

        # 3. Filter disqualified pairs
        disqualified = [e for e in evaluations if e.disqualified]
        valid = [e for e in evaluations if not e.disqualified]

        if disqualified:
            # Log disqualification summary so operators can diagnose
            reason_counts: Dict[str, int] = {}
            for e in disqualified:
                reason_counts[e.disqualify_reason] = reason_counts.get(e.disqualify_reason, 0) + 1
            logger.info(
                f"Disqualified {len(disqualified)}/{len(evaluations)} pairs: "
                + ", ".join(f"{r}={c}" for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]))
            )
            # Log top 3 closest-to-qualifying pairs for debugging
            scored_dq = sorted(disqualified, key=lambda e: e.prediction.get("confidence", 0), reverse=True)
            for e in scored_dq[:3]:
                conf = e.prediction.get("confidence", 0)
                logger.debug(
                    f"  Near-miss: {e.pair} confidence={conf:.3f} reason={e.disqualify_reason}"
                )

        if not valid:
            logger.info("All pairs disqualified – no valid signals this cycle")
            return None

        # 4. Select best pair
        valid.sort(key=lambda e: e.score, reverse=True)
        best = valid[0]

        logger.info(
            f"Best pair: {best.pair} | Score: {best.score:.3f} | "
            f"Confidence: {best.prediction.get('confidence', 0):.3f}"
        )

        # 5. Generate trade signal
        signal = self._build_signal(best)
        if signal is None:
            logger.info("Signal generation failed for best pair")
            return None

        return signal

    # ------------------------------------------------------------------
    # Pair selection
    # ------------------------------------------------------------------

    def _get_candidate_pairs(self) -> List[Dict[str, Any]]:
        """Get top USDT pairs by volume."""
        try:
            pairs = self.fetcher.get_top_usdt_pairs(settings.trading.top_pairs_count)
            logger.info(f"Found {len(pairs)} candidate pairs")
            return pairs
        except Exception as e:
            logger.error(f"Error fetching pairs: {e}")
            return []

    def _evaluate_pair(
        self,
        pair: str,
        pair_info: Dict[str, Any],
        btc_df: Optional[pd.DataFrame] = None,
    ) -> PairEvaluation:
        """Evaluate a single pair for trading suitability."""
        reasons: List[str] = []
        score = 0.0

        # Fetch recent data — request enough history for features + encoder
        # encoder_length (96) + warmup for indicators (~200) ≈ need ~300 candles
        min_rows = settings.model.encoder_length + 110  # ~206 rows minimum
        df = self.fetcher.fetch_klines(
            pair, "15min",
            start_dt=datetime.utcnow() - timedelta(hours=80),
        )
        if df.empty or len(df) < min_rows:
            return PairEvaluation(
                pair=pair, score=0, prediction={}, features={},
                reasons=[f"Insufficient data ({len(df) if not df.empty else 0} rows, need {min_rows})"],
                disqualified=True,
                disqualify_reason="Insufficient historical data",
            )

        # Compute features (use pre-fetched BTC data)
        df["pair"] = pair
        df = compute_features(df, btc_df)

        # Run prediction
        prediction = self.predictor.predict(df, pair)

        # Feature snapshot (latest row)
        latest = df.iloc[-1]
        features = {
            "atr_14": float(latest.get("atr_14", 0)),
            "rsi_14": float(latest.get("rsi_14", 50)),
            "volatility_20": float(latest.get("volatility_20", 0)),
            "volatility_regime": str(latest.get("volatility_regime", "unknown")),
            "market_regime": str(latest.get("market_regime", "unknown")),
            "volume_ratio": float(latest.get("volume_ratio", 0)),
            "btc_correlation": float(latest.get("btc_correlation", 0)),
            "bb_position": float(latest.get("bb_position", 0.5)),
            "macd_hist": float(latest.get("macd_hist", 0)),
            "momentum_10": float(latest.get("momentum_10", 0)),
        }

        # ---- Confidence check ----
        confidence = prediction.get("confidence", 0)
        if confidence < self.confidence_threshold:
            return PairEvaluation(
                pair=pair, score=0, prediction=prediction, features=features,
                reasons=[f"Confidence {confidence:.3f} < threshold {self.confidence_threshold}"],
                disqualified=True,
                disqualify_reason="Below confidence threshold",
            )

        # Score: confidence weight (40%)
        score += confidence * 0.40
        reasons.append(f"Confidence: {confidence:.3f}")

        # ---- Volatility check ----
        vol_regime = features["volatility_regime"]
        if vol_regime == "extreme":
            return PairEvaluation(
                pair=pair, score=0, prediction=prediction, features=features,
                reasons=["Extreme volatility"], disqualified=True,
                disqualify_reason="Extreme volatility regime",
            )
        if vol_regime == "low":
            score += 0.05
            reasons.append("Low volatility (good for controlled trades)")
        elif vol_regime == "normal":
            score += 0.10
            reasons.append("Normal volatility")
        elif vol_regime == "high":
            score += 0.03
            reasons.append("High volatility (caution)")

        # ---- ATR filter ----
        atr = features["atr_14"]
        price = float(latest["close"])
        atr_pct = atr / price if price > 0 else 0
        if atr_pct < 0.001:  # Less than 0.1% ATR
            reasons.append("ATR too low - insufficient movement expected")
            score -= 0.05

        # ---- Spread check ----
        try:
            _, spread_pct = self.fetcher.get_spread(pair)
            if spread_pct > self._spread_max_pct:
                return PairEvaluation(
                    pair=pair, score=0, prediction=prediction, features=features,
                    reasons=[f"Spread too wide: {spread_pct:.4f}"],
                    disqualified=True, disqualify_reason="Spread too wide",
                )
            score += (1 - spread_pct / self._spread_max_pct) * 0.10
            reasons.append(f"Spread: {spread_pct:.4f}")
        except Exception:
            spread_pct = 0.001
            reasons.append("Could not check spread")

        # ---- Volume check ----
        vol_ratio = features["volume_ratio"]
        if vol_ratio < self._min_volume_ratio:
            reasons.append(f"Low volume ratio: {vol_ratio:.2f}")
            score -= 0.05
        else:
            score += min(vol_ratio * 0.05, 0.15)
            reasons.append(f"Volume ratio: {vol_ratio:.2f}")

        # ---- Expected move magnitude ----
        expected_move = prediction.get("expected_move", 0)
        move_score = min(abs(expected_move) * 100, 0.20)
        score += move_score
        reasons.append(f"Expected move: {expected_move:.4f}")

        # ---- BTC correlation adjustment ----
        btc_corr = features["btc_correlation"]
        # High correlation means pair is BTC-dependent (less alpha)
        if abs(btc_corr) > 0.9:
            score -= 0.05
            reasons.append(f"Very high BTC correlation: {btc_corr:.2f}")

        # ---- Risk-reward estimate ----
        # Quick R:R check
        if prediction.get("prob_up", 0) > prediction.get("prob_down", 0):
            rr_estimate = abs(expected_move) / (atr_pct * 2) if atr_pct > 0 else 0
        else:
            rr_estimate = 0  # We only go long in spot
        if rr_estimate > 2:
            score += 0.10
            reasons.append(f"Good R:R estimate: {rr_estimate:.2f}")

        # Store volume for signal
        vol_24h = float(pair_info.get("volValue", 0))

        return PairEvaluation(
            pair=pair,
            score=max(score, 0),
            prediction={**prediction, "spread_pct": spread_pct, "volume_24h": vol_24h},
            features=features,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(self, evaluation: PairEvaluation) -> Optional[TradeSignal]:
        """Build a complete trade signal from an evaluation."""
        pred = evaluation.prediction
        feat = evaluation.features
        pair = evaluation.pair

        # Only long trades in spot
        if pred.get("prob_up", 0) <= pred.get("prob_down", 0):
            logger.info(f"{pair}: Bearish signal, skipping (spot only)")
            return None

        try:
            ticker = self.fetcher.get_ticker(pair)
            entry_price = ticker["best_ask"]  # Buy at ask
        except Exception as e:
            logger.error(f"Cannot get ticker for {pair}: {e}")
            return None

        if entry_price <= 0:
            return None

        # ATR-based stop loss (2x ATR below entry)
        atr = feat.get("atr_14", entry_price * 0.01)
        stop_distance = atr * 2.0
        stop_price = entry_price - stop_distance

        # Target: based on expected move + R:R minimum of 2:1
        expected_move_abs = abs(pred.get("expected_move", 0)) * entry_price
        min_target_distance = stop_distance * 2.0  # At least 2R
        target_distance = max(expected_move_abs, min_target_distance)
        target_price = entry_price + target_distance

        risk_reward = target_distance / stop_distance if stop_distance > 0 else 0

        # Build reasoning
        reasoning = self._build_reasoning(evaluation, entry_price, stop_price, target_price, risk_reward)

        return TradeSignal(
            pair=pair,
            side="BUY",
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            confidence=pred.get("confidence", 0),
            expected_move=pred.get("expected_move", 0),
            prob_up=pred.get("prob_up", 0),
            prob_down=pred.get("prob_down", 0),
            risk_reward=risk_reward,
            atr=atr,
            volatility_regime=feat.get("volatility_regime", "unknown"),
            market_regime=feat.get("market_regime", "unknown"),
            btc_correlation=feat.get("btc_correlation", 0),
            spread_pct=pred.get("spread_pct", 0),
            volume_24h=pred.get("volume_24h", 0),
            reasoning=reasoning,
            features_snapshot=feat,
            forecast_vector=pred.get("forecast_vector", []),
        )

    def _build_reasoning(
        self,
        eval: PairEvaluation,
        entry: float,
        stop: float,
        target: float,
        rr: float,
    ) -> str:
        """Build human-readable AI reasoning for the trade."""
        lines = [
            f"Pair: {eval.pair}",
            f"Score: {eval.score:.3f}",
            f"Direction: LONG (prob_up={eval.prediction.get('prob_up', 0):.3f})",
            f"Entry: {entry:.6f} | Stop: {stop:.6f} | Target: {target:.6f}",
            f"Risk/Reward: {rr:.2f}R",
            f"Confidence: {eval.prediction.get('confidence', 0):.3f}",
            f"Market Regime: {eval.features.get('market_regime', 'unknown')}",
            f"Volatility Regime: {eval.features.get('volatility_regime', 'unknown')}",
            f"BTC Correlation: {eval.features.get('btc_correlation', 0):.3f}",
            "",
            "Evaluation Factors:",
        ]
        for reason in eval.reasons:
            lines.append(f"  • {reason}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dynamic threshold adjustments (called by self-improving loop)
    # ------------------------------------------------------------------

    def update_thresholds(self, adjustments: Dict[str, float]) -> None:
        """Update decision thresholds based on feedback analysis."""
        if "confidence_threshold" in adjustments:
            old = self.confidence_threshold
            self.confidence_threshold = max(0.5, min(0.95, adjustments["confidence_threshold"]))
            logger.info(f"Confidence threshold: {old:.3f} → {self.confidence_threshold:.3f}")

        if "spread_max_pct" in adjustments:
            self._spread_max_pct = adjustments["spread_max_pct"]

        if "min_volume_ratio" in adjustments:
            self._min_volume_ratio = adjustments["min_volume_ratio"]

        if "risk_per_trade" in adjustments:
            self.risk_per_trade = max(0.005, min(0.03, adjustments["risk_per_trade"]))

    def get_current_thresholds(self) -> Dict[str, float]:
        """Return current decision thresholds."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "risk_per_trade": self.risk_per_trade,
            "spread_max_pct": self._spread_max_pct,
            "min_volume_ratio": self._min_volume_ratio,
            "btc_corr_weight": self._btc_corr_weight,
        }
