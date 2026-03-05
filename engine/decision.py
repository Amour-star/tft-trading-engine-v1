"""
AI Decision Engine.
Evaluates TFT predictions, selects optimal pairs, and generates trade signals.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import ACTIVE_SYMBOL, TRADING_UNIVERSE, XRP_ONLY_SYMBOL, settings
from data.database import DecisionEvent, PerformanceMetric, Prediction, SignalRecord, Trade, get_session
from data.fetcher import KuCoinDataFetcher
from data.features import compute_features
from engine.adaptive_threshold import AdaptiveConfidenceThreshold
from models.meta_model import XGBoostMetaModel
from models.tft_model import TFTPredictor
from services.regime_engine import RegimeEngine


@dataclass
class TradeSignal:
    """A fully evaluated trade signal."""
    pair: str
    side: str  # BUY for long, SELL for short
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
    position_size_multiplier: float = 1.0
    model_entry_price: float = 0.0
    market_mark_price: float = 0.0
    market_entry_price: float = 0.0
    adaptive_threshold: float = 0.0
    regime: Dict[str, Any] = field(default_factory=dict)
    features_snapshot: Dict[str, Any] = field(default_factory=dict)
    forecast_vector: List[float] = field(default_factory=list)
    ai_score: float = 0.0
    base_ai_score: float = 0.0
    ai_confidence: float = 0.0
    meta_probability: float = 0.0
    meta_model_version: str = ""
    tft_model_version: str = ""
    governance_code: str = ""
    governance_approved: bool = True
    governance_size_mult: float = 1.0
    governance_conf_adj: float = 0.0
    governance_risk_mode: str = "neutral"
    tft_score: float = 0.0
    xgb_score: float = 0.0
    ppo_score: float = 0.0
    gov_adjust: float = 0.0
    final_ai_score: float = 0.0
    weight_snapshot: Dict[str, Any] = field(default_factory=dict)
    risk_per_trade: float = 0.01
    signal_score: float = 0.5
    momentum_factor: float = 0.5
    volatility_factor: float = 0.5
    trend_factor: float = 0.5
    mean_reversion_factor: float = 0.5
    volume_factor: float = 0.5
    volume_imbalance: float = 0.0
    decision_label: str = "HOLD"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PairEvaluation:
    """Evaluation result for a single pair."""
    pair: str
    score: float
    prediction: Dict[str, Any]
    features: Dict[str, Any]
    reasons: List[str]
    meta_prediction: Dict[str, Any] = field(default_factory=dict)
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
        self.meta_model = XGBoostMetaModel()
        self.confidence_threshold = settings.trading.confidence_threshold
        self.min_confidence = float(
            os.getenv("MIN_CONFIDENCE", str(self.confidence_threshold))
        )
        self.allow_shorts = bool(settings.trading.allow_shorts and not settings.trading.spot_only_mode)
        self.aggression_level = float(settings.trading.aggression_level)
        self.regime_engine = RegimeEngine()
        self.adaptive_threshold = AdaptiveConfidenceThreshold(
            base_threshold=self.min_confidence,
            min_threshold=0.40,
            max_threshold=0.80,
            aggression_level=self.aggression_level,
        )
        self.risk_per_trade = settings.trading.risk_per_trade
        self.tft_weight = 0.40
        self.xgb_weight = 0.40
        self.ppo_weight = 0.20
        self._normalize_agent_weights()
        self._tft_disable_flag_path = Path(
            os.getenv("TFT_DISABLE_FLAG_PATH", "/app/state/tft_disabled.flag")
        )
        self._tft_disable_warned = False
        if settings.trading.spot_only_mode:
            logger.info("Spot-only mode enabled: short signals are disabled")

        # Dynamic thresholds (adjusted by self-improving loop)
        self._atr_min_multiplier: float = 1.0
        self._spread_max_pct: float = float(settings.trading.max_spread_pct)
        self._min_volume_ratio: float = 0.5
        self._btc_corr_weight: float = 0.15
        self._factor_weights: Dict[str, float] = {
            "momentum": 0.28,
            "volatility": 0.16,
            "trend": 0.24,
            "mean_reversion": 0.14,
            "volume": 0.18,
        }

    # ------------------------------------------------------------------
    # Main signal generation pipeline
    # ------------------------------------------------------------------

    def generate_signal(self) -> Optional[TradeSignal]:
        """
        Full pipeline: scan pairs → evaluate → select best → generate signal.
        Returns None if no valid signal found.
        """
        fallback_mode = self._is_tft_fallback_mode()
        logger.bind(
            event="DECISION_CYCLE_STARTED",
            symbol=ACTIVE_SYMBOL,
            fallback_mode=fallback_mode,
            model_loaded=self.predictor.model is not None,
            model_version=self.predictor.model_version or "none",
        ).info("DECISION_CYCLE_STARTED")

        # 1. Get candidate pairs
        candidates = self._get_candidate_pairs()
        if not candidates:
            logger.info("No candidate pairs found")
            self._record_no_trade_event(evaluations=[], valid=[], reason="no_candidate_pairs")
            return None

        try:
            supported_pairs = self._get_supported_pairs()
        except Exception as exc:
            logger.error(f"Unable to inspect model pair vocabulary: {exc}")
            self._record_no_trade_event(
                evaluations=[],
                valid=[],
                reason="model_vocabulary_unavailable",
            )
            return None

        supported_list = sorted(supported_pairs)
        logger.info(f"Model supports: {supported_list}")
        if ACTIVE_SYMBOL not in supported_pairs:
            logger.error(
                f"Model vocabulary excludes {ACTIVE_SYMBOL} while engine is locked to this symbol"
            )
            self._record_no_trade_event(
                evaluations=[],
                valid=[],
                reason=f"model_missing_{ACTIVE_SYMBOL}_vocabulary",
            )
            return None

        # 2. Evaluate each pair
        evaluations: List[PairEvaluation] = []
        for pair_info in candidates:
            pair = pair_info["symbol"]
            try:
                evaluation = self._evaluate_pair(pair, pair_info, btc_df=None)
                evaluations.append(evaluation)
            except Exception as e:
                logger.exception(f"Error evaluating {pair}: {e}")

        # 3. Filter disqualified pairs
        disqualified = [e for e in evaluations if e.disqualified]
        valid = [e for e in evaluations if not e.disqualified]

        for ev in evaluations:
            logger.bind(
                event="CANDIDATE_GENERATED",
                pair=ev.pair,
                score=round(ev.score, 4),
                disqualified=ev.disqualified,
                disqualify_reason=ev.disqualify_reason or None,
                fallback_mode=fallback_mode,
            ).info("CANDIDATE_GENERATED")

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
            scored_dq = sorted(
                disqualified,
                key=lambda e: e.prediction.get("meta_probability", e.prediction.get("confidence", 0)),
                reverse=True,
            )
            for e in scored_dq[:3]:
                conf = e.prediction.get("meta_probability", e.prediction.get("confidence", 0))
                logger.debug(
                    f"  Near-miss: {e.pair} confidence={conf:.3f} reason={e.disqualify_reason}"
                )

        if not valid:
            logger.info("All pairs disqualified – no valid signals this cycle")
            self._record_no_trade_event(
                evaluations=evaluations,
                valid=[],
                reason="all_pairs_disqualified",
            )
            return None

        # 4. Sort candidates by score (best first)
        valid.sort(key=lambda e: e.score, reverse=True)

        # 5. Iterate candidates and try to build a signal.
        skip_reasons: List[str] = []
        for candidate in valid:
            logger.info(
                f"Candidate: {candidate.pair} | Score: {candidate.score:.3f} | "
                f"AI Score: {candidate.prediction.get('meta_probability', candidate.prediction.get('confidence', 0)):.3f}"
            )
            signal = self._build_signal(candidate)
            if signal is not None:
                return signal
            # Record why this candidate was skipped
            skip_reasons.append(
                f"{candidate.pair}(score={candidate.score:.3f},"
                f"prob_up={candidate.prediction.get('prob_up', 0):.3f},"
                f"prob_down={candidate.prediction.get('prob_down', 0):.3f})"
            )

        # All valid candidates were skipped by final signal guards.
        logger.info(
            f"No tradable signal found among {len(valid)} valid candidates: "
            f"{', '.join(skip_reasons[:5])}"
        )

        # Persist a NO_TRADE decision event for dashboard visibility
        self._record_no_trade_event(
            evaluations=evaluations,
            valid=valid,
            reason="no_trade_after_signal_build",
        )

        return None

    # ------------------------------------------------------------------
    # Pair selection
    # ------------------------------------------------------------------

    def _get_candidate_pairs(self) -> List[Dict[str, Any]]:
        """Return the single symbol assigned to this engine instance."""
        logger.info(f"Universe locked to {ACTIVE_SYMBOL}")
        logger.info(f"Universe size: 1 | {ACTIVE_SYMBOL}")
        return [{"symbol": ACTIVE_SYMBOL, "volValue": "0"}]

    def _get_supported_pairs(self) -> set[str]:
        """
        Return model-supported pairs if predictor exposes vocabulary introspection.
        In fallback mode, returns the active symbol as supported.
        """
        if self._is_tft_fallback_mode():
            return {ACTIVE_SYMBOL}

        getter = getattr(self.predictor, "get_supported_pairs", None)
        if not callable(getter):
            logger.warning(f"Predictor does not expose get_supported_pairs(); assuming {ACTIVE_SYMBOL} supported")
            return {ACTIVE_SYMBOL}

        supported_raw = getter()
        if not isinstance(supported_raw, (list, tuple, set)):
            logger.warning(f"Predictor returned invalid pair vocabulary; assuming {ACTIVE_SYMBOL} supported")
            return {ACTIVE_SYMBOL}

        supported = {str(item).strip() for item in supported_raw if str(item).strip()}
        if not supported:
            logger.warning(f"Predictor pair vocabulary is empty; assuming {ACTIVE_SYMBOL} supported")
            return {ACTIVE_SYMBOL}
        return supported

    def _evaluate_pair(
        self,
        pair: str,
        pair_info: Dict[str, Any],
        btc_df: Optional[pd.DataFrame] = None,
    ) -> PairEvaluation:
        """Evaluate a single pair for trading suitability."""
        reasons: List[str] = []
        score = 0.0

        supported_pairs = self._get_supported_pairs()
        if supported_pairs is not None and pair not in supported_pairs:
            return PairEvaluation(
                pair=pair,
                score=0.0,
                prediction={},
                features={},
                reasons=[f"Pair {pair} not in trained model vocabulary"],
                disqualified=True,
                disqualify_reason="Unsupported model pair",
            )

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

        latest_ts = pd.to_datetime(df["timestamp"].iloc[-1], errors="coerce", utc=True)
        if pd.isna(latest_ts):
            return PairEvaluation(
                pair=pair,
                score=0.0,
                prediction={},
                features={},
                reasons=["Latest candle timestamp is invalid"],
                disqualified=True,
                disqualify_reason="Invalid candle timestamp",
            )

        age_seconds = (datetime.now(timezone.utc) - latest_ts.to_pydatetime()).total_seconds()
        stale_threshold = 2 * 15 * 60
        if age_seconds > stale_threshold:
            logger.bind(
                event="STALE_DATA_GUARD",
                pair=pair,
                age_seconds=round(age_seconds, 1),
                threshold_seconds=stale_threshold,
                last_candle=latest_ts.isoformat(),
            ).warning("STALE_DATA_GUARD")
            return PairEvaluation(
                pair=pair,
                score=0.0,
                prediction={},
                features={},
                reasons=[
                    f"Stale candles detected ({age_seconds:.1f}s old > {stale_threshold}s threshold)"
                ],
                disqualified=True,
                disqualify_reason="Stale market data",
            )

        # Compute features (use pre-fetched BTC data)
        df["pair"] = pair
        df = compute_features(df, btc_df)
        regime_info = self.regime_engine.classify(df)

        # Run prediction
        prediction = self.predictor.predict(df, pair)
        if not bool(prediction.get("valid", True)):
            if self._is_tft_fallback_mode():
                prediction = self._fallback_prediction_payload()
                reasons.append("TFT disabled, using xgb_meta_fallback mode")
            else:
                return PairEvaluation(
                    pair=pair,
                    score=0.0,
                    prediction=prediction,
                    features={},
                    reasons=["Invalid TFT forecast (NaN/Inf or empty output)"],
                    disqualified=True,
                    disqualify_reason="Invalid prediction",
                )

        expected_move = float(prediction.get("expected_move", 0.0))
        if not np.isfinite(expected_move):
            logger.warning(f"{pair}: non-finite expected_move detected; coercing to 0.0")
            expected_move = 0.0
        prediction["expected_move"] = expected_move

        # Feature snapshot (latest row)
        latest = df.iloc[-1]
        features = {
            "model_close_price": float(latest.get("close", 0.0)),
            "atr_14": float(latest.get("atr_14", 0)),
            "rsi_14": float(latest.get("rsi_14", 50)),
            "volatility_20": float(latest.get("volatility_20", 0)),
            "volatility_regime": str(regime_info.get("volatility", latest.get("volatility_regime", "unknown"))),
            "market_regime": str(regime_info.get("trend", latest.get("market_regime", "unknown"))),
            "momentum_regime": str(regime_info.get("momentum", "weak")),
            "regime_score": float(regime_info.get("regime_score", 0.0)),
            "volume_ratio": float(latest.get("volume_ratio", 0)),
            "btc_correlation": float(latest.get("btc_correlation", 0)),
            "bb_position": float(latest.get("bb_position", 0.5)),
            "macd_hist": float(latest.get("macd_hist", 0)),
            "momentum_10": float(latest.get("momentum_10", 0)),
            "hour_sin": float(latest.get("hour_sin", 0)),
            "hour_cos": float(latest.get("hour_cos", 0)),
            "dow_sin": float(latest.get("dow_sin", 0)),
            "dow_cos": float(latest.get("dow_cos", 0)),
        }

        ts_value = latest.get("timestamp", datetime.utcnow())
        if isinstance(ts_value, pd.Timestamp):
            ts_value = ts_value.to_pydatetime()
        elif not isinstance(ts_value, datetime):
            ts_value = datetime.utcnow()

        meta_features = self.meta_model.build_features(prediction, features, ts_value)
        meta_prediction = self.meta_model.predict(prediction, meta_features)
        prediction = {
            **prediction,
            "meta_probability": float(meta_prediction.probability),
            "meta_confidence": float(meta_prediction.confidence_score),
            "meta_model_version": str(meta_prediction.model_version),
            "meta_features": meta_features,
        }

        performance_metrics = self._get_performance_metrics()
        adaptive_base = self.adaptive_threshold.compute(performance_metrics, regime_info)
        adaptive_threshold = self.regime_engine.scale_threshold_by_regime(
            adaptive_base,
            regime_info,
            allow_shorts=self.allow_shorts,
        )
        regime_size_multiplier = self.regime_engine.position_size_multiplier(regime_info)
        logger.info(
            f"Adaptive threshold: {adaptive_threshold:.3f} "
            f"(base={adaptive_base:.3f}, aggr={self.aggression_level:.2f}) | Regime: {regime_info}"
        )
        prediction["adaptive_threshold"] = float(adaptive_threshold)
        prediction["regime_info"] = regime_info
        prediction["regime_size_multiplier"] = float(regime_size_multiplier)

        confidence = float(prediction.get("confidence", 0.0))
        logger.bind(
            event="PAIR_EVAL",
            pair=pair,
            confidence=confidence,
            threshold=float(adaptive_threshold),
            regime_score=float(regime_info.get("regime_score", 0.0)),
            trend=regime_info.get("trend", "unknown"),
            volatility=regime_info.get("volatility", "unknown"),
            momentum=regime_info.get("momentum", "unknown"),
        ).info("PAIR_EVAL")

        self._record_prediction(pair, prediction, features)

        multifactor = self._compute_multifactor_components(pair, latest, df)
        prediction["signal_score"] = float(multifactor["signal_score"])
        prediction["momentum_factor"] = float(multifactor["momentum_factor"])
        prediction["volatility_factor"] = float(multifactor["volatility_factor"])
        prediction["trend_factor"] = float(multifactor["trend_factor"])
        prediction["mean_reversion_factor"] = float(multifactor["mean_reversion_factor"])
        prediction["volume_factor"] = float(multifactor["volume_factor"])
        prediction["volume_imbalance"] = float(multifactor["volume_imbalance"])
        score += float(multifactor["signal_score"]) * 0.35
        reasons.append(
            "Multi-factor score: "
            f"{float(multifactor['signal_score']):.3f} "
            f"(mom={float(multifactor['momentum_factor']):.3f}, "
            f"vol={float(multifactor['volatility_factor']):.3f}, "
            f"trend={float(multifactor['trend_factor']):.3f}, "
            f"mr={float(multifactor['mean_reversion_factor']):.3f}, "
            f"volume={float(multifactor['volume_factor']):.3f}, "
            f"imbalance={float(multifactor['volume_imbalance']):.3f})"
        )

        # ---- Confidence check (aggressive mode: base threshold 0.45) ----
        ai_score = float(prediction.get("meta_probability", 0.0))
        aggressive_threshold = 0.45
        if ai_score < aggressive_threshold:
            return PairEvaluation(
                pair=pair, score=0, prediction=prediction, features=features,
                reasons=[f"Meta probability {ai_score:.3f} < aggressive threshold {aggressive_threshold:.3f}"],
                disqualified=True,
                disqualify_reason="Below confidence threshold",
            )
        prediction["size_scale"] = 1.0
        reasons.append(f"Confidence {ai_score:.3f} >= {aggressive_threshold:.3f} threshold")

        tft_confidence = float(prediction.get("confidence", 0.0))

        # Dynamic base score: XGB and TFT blend from current strategy weights.
        xgb_w = self.xgb_weight
        tft_w = self.tft_weight
        blend_total = xgb_w + tft_w
        if blend_total <= 0:
            xgb_w = 0.5
            tft_w = 0.5
            blend_total = 1.0
        score += ai_score * (xgb_w / blend_total)
        reasons.append(f"AI score (XGB): {ai_score:.3f} (w={xgb_w / blend_total:.2f})")
        score += tft_confidence * (tft_w / blend_total)
        reasons.append(f"TFT confidence: {tft_confidence:.3f} (w={tft_w / blend_total:.2f})")
        score += min(max(features.get("regime_score", 0.0), 0.0), 1.0) * 0.05
        reasons.append(f"Regime score: {features.get('regime_score', 0.0):.3f}")

        # ---- Volatility check (aggressive: regime modifies size, never blocks) ----
        vol_regime = features["volatility_regime"]
        if vol_regime == "extreme":
            score += 0.01
            prediction["size_scale"] = 0.5  # Reduce size but don't block
            reasons.append("Extreme volatility (reduced position size)")
        elif vol_regime == "low":
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
        expected_move = float(prediction.get("expected_move", 0.0))
        if not np.isfinite(expected_move):
            expected_move = 0.0
            prediction["expected_move"] = 0.0
        move_score = min(abs(expected_move) * 100, 0.20)
        score += move_score
        reasons.append(f"Expected move: {expected_move:.4f}")

        # ---- Benchmark correlation adjustment ----
        btc_corr = features["btc_correlation"]
        # High correlation means pair is benchmark-dependent (less alpha).
        if abs(btc_corr) > 0.9:
            score -= 0.05
            reasons.append(f"Very high benchmark correlation: {btc_corr:.2f}")

        # ---- Risk-reward estimate ----
        # Quick R:R check.
        bullish = float(prediction.get("prob_up", 0.0)) > float(prediction.get("prob_down", 0.0))
        if bullish or self.allow_shorts:
            rr_estimate = abs(expected_move) / (atr_pct * 2) if atr_pct > 0 else 0
        else:
            rr_estimate = 0
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
            meta_prediction={
                "probability": ai_score,
                "confidence_score": prediction.get("meta_confidence", 0),
                "model_version": prediction.get("meta_model_version", ""),
            },
        )

    # ------------------------------------------------------------------
    # Signal building
    # ------------------------------------------------------------------

    def _build_signal(self, evaluation: PairEvaluation) -> Optional[TradeSignal]:
        """Build a complete trade signal from an evaluation."""
        pred = evaluation.prediction
        feat = evaluation.features
        pair = evaluation.pair

        prob_up = float(pred.get("prob_up", 0.0))
        prob_down = float(pred.get("prob_down", 0.0))
        signal_score = self._bounded(pred.get("signal_score", 0.5))
        if signal_score >= 0.58:
            decision_label = "LONG"
        elif signal_score <= 0.42:
            decision_label = "SHORT"
        else:
            if prob_up - prob_down >= 0.05:
                decision_label = "LONG"
            elif prob_down - prob_up >= 0.05:
                decision_label = "SHORT"
            else:
                decision_label = "HOLD"

        if decision_label == "SHORT" and not self.allow_shorts:
            logger.info(f"{pair}: Bearish signal skipped (ALLOW_SHORTS disabled)")
            self._record_signal_record(
                pair=pair,
                signal_score=signal_score,
                decision="HOLD",
                side="HOLD",
                prediction=pred,
                reason="short_disabled",
            )
            return None

        if decision_label == "HOLD":
            self._record_signal_record(
                pair=pair,
                signal_score=signal_score,
                decision="HOLD",
                side="HOLD",
                prediction=pred,
                reason="neutral_score",
            )
            return None

        direction = decision_label
        side = "SELL" if decision_label == "SHORT" else "BUY"

        try:
            ticker = self.fetcher.get_ticker(pair)
            mark_price = float(ticker.get("price") or 0.0)
            entry_price = mark_price
        except Exception as e:
            logger.error(f"Cannot get ticker for {pair}: {e}")
            return None

        if entry_price <= 0:
            return None

        model_entry_price = float(feat.get("model_close_price", 0.0) or 0.0)
        if model_entry_price <= 0:
            model_entry_price = float(entry_price)
        if mark_price <= 0:
            mark_price = float(entry_price)

        # ATR-based stop loss (2x ATR away from entry).
        atr = float(feat.get("atr_14", entry_price * 0.01))
        stop_distance = atr * 2.0
        if side == "BUY":
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        # Target: based on expected move + R:R minimum of 2:1.
        expected_move_abs = abs(float(pred.get("expected_move", 0.0))) * entry_price
        min_target_distance = stop_distance * 2.0  # At least 2R
        target_distance = max(expected_move_abs, min_target_distance)
        if side == "BUY":
            target_price = entry_price + target_distance
        else:
            target_price = entry_price - target_distance

        risk_reward = target_distance / stop_distance if stop_distance > 0 else 0

        # Build reasoning.
        reasoning = self._build_reasoning(
            evaluation,
            entry_price,
            stop_price,
            target_price,
            risk_reward,
            direction=direction,
        )

        tft_score = float(pred.get("confidence", 0.0))
        xgb_score = float(pred.get("meta_probability", 0.0))
        blend_weight = self.tft_weight + self.xgb_weight
        if blend_weight <= 0:
            base_score = max(0.0, min(1.0, (tft_score + xgb_score) / 2.0))
        else:
            base_score = max(
                0.0,
                min(
                    1.0,
                    (tft_score * self.tft_weight + xgb_score * self.xgb_weight) / blend_weight,
                ),
            )

        regime_score = float(feat.get("regime_score", 0.0))
        adaptive_threshold = float(
            pred.get(
                "adaptive_threshold",
                max(self.confidence_threshold, getattr(self, "min_confidence", self.confidence_threshold)),
            )
        )
        # Aggressive mode: regime score does NOT block trades, only logged
        if regime_score < 0.10:
            logger.info(
                "Low regime score {regime_score:.3f} (confidence={confidence:.3f}) - proceeding anyway (aggressive mode)",
                regime_score=regime_score,
                confidence=base_score,
            )

        size_scale = float(pred.get("size_scale", 1.0))
        regime_size_multiplier = float(pred.get("regime_size_multiplier", 1.0))
        position_size_multiplier = max(0.25, min(2.0, size_scale * regime_size_multiplier))
        signal = TradeSignal(
            pair=pair,
            side=side,
            entry_price=entry_price,
            model_entry_price=model_entry_price,
            market_mark_price=mark_price,
            market_entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            confidence=base_score,
            expected_move=float(pred.get("expected_move", 0.0)),
            prob_up=prob_up,
            prob_down=prob_down,
            risk_reward=risk_reward,
            atr=atr,
            volatility_regime=feat.get("volatility_regime", "unknown"),
            market_regime=feat.get("market_regime", "unknown"),
            btc_correlation=feat.get("btc_correlation", 0),
            spread_pct=pred.get("spread_pct", 0),
            volume_24h=pred.get("volume_24h", 0),
            reasoning=reasoning,
            position_size_multiplier=position_size_multiplier,
            adaptive_threshold=adaptive_threshold,
            regime=dict(pred.get("regime_info", {})),
            features_snapshot=feat,
            forecast_vector=pred.get("forecast_vector", []),
            ai_score=base_score,
            base_ai_score=base_score,
            ai_confidence=pred.get("meta_confidence", 0),
            meta_probability=xgb_score,
            meta_model_version=pred.get("meta_model_version", ""),
            tft_model_version=pred.get("model_version", ""),
            tft_score=tft_score,
            xgb_score=xgb_score,
            ppo_score=0.0,
            gov_adjust=0.0,
            final_ai_score=base_score,
            weight_snapshot={
                "tft_weight": self.tft_weight,
                "xgb_weight": self.xgb_weight,
                "ppo_weight": self.ppo_weight,
            },
            risk_per_trade=self.risk_per_trade,
            signal_score=signal_score,
            momentum_factor=self._bounded(pred.get("momentum_factor", 0.5)),
            volatility_factor=self._bounded(pred.get("volatility_factor", 0.5)),
            trend_factor=self._bounded(pred.get("trend_factor", 0.5)),
            mean_reversion_factor=self._bounded(pred.get("mean_reversion_factor", 0.5)),
            volume_factor=self._bounded(pred.get("volume_factor", 0.5)),
            volume_imbalance=float(pred.get("volume_imbalance", 0.0) or 0.0),
            decision_label=direction,
        )
        self._record_signal_record(
            pair=pair,
            signal_score=signal.signal_score,
            decision=direction,
            side=side,
            prediction=pred,
            reason="signal_ready",
        )
        return signal

    def _is_tft_fallback_mode(self) -> bool:
        """Return True if we should use XGB fallback instead of TFT.
        This is True when: model is not loaded, or TFT_FORCE_DISABLE is set."""
        if self.predictor.model is None:
            if not self._tft_disable_warned:
                logger.warning("No TFT model loaded; using xgb_meta_fallback mode.")
                self._tft_disable_warned = True
            return True
        force_disable = os.getenv("TFT_FORCE_DISABLE", "").strip().lower() in ("true", "1", "yes")
        if force_disable:
            if not self._tft_disable_warned:
                logger.warning("TFT_FORCE_DISABLE set; using xgb_meta_fallback mode.")
                self._tft_disable_warned = True
            return True
        return False

    @staticmethod
    def _fallback_prediction_payload() -> Dict[str, Any]:
        return {
            "prob_up": 0.5,
            "prob_down": 0.5,
            "expected_move": 0.0,
            "confidence": 0.5,
            "valid": True,
            "forecast_vector": [0.0] * 12,
            "lower_bound": [0.0] * 12,
            "upper_bound": [0.0] * 12,
            "attention_stats": {
                "mean": 0.0,
                "std": 0.0,
                "peak": 0.0,
                "consistency": 0.0,
            },
            "model_version": "tft_disabled",
        }

    def _build_reasoning(
        self,
        eval: PairEvaluation,
        entry: float,
        stop: float,
        target: float,
        rr: float,
        direction: str,
    ) -> str:
        """Build human-readable AI reasoning for the trade."""
        lines = [
            f"Pair: {eval.pair}",
            f"Score: {eval.score:.3f}",
            f"Direction: {direction} (prob_up={eval.prediction.get('prob_up', 0):.3f}, prob_down={eval.prediction.get('prob_down', 0):.3f})",
            f"Entry: {entry:.6f} | Stop: {stop:.6f} | Target: {target:.6f}",
            f"Risk/Reward: {rr:.2f}R",
            f"TFT Confidence: {eval.prediction.get('confidence', 0):.3f}",
            f"AI Score (XGB): {eval.prediction.get('meta_probability', 0):.3f}",
            f"Market Regime: {eval.features.get('market_regime', 'unknown')}",
            f"Volatility Regime: {eval.features.get('volatility_regime', 'unknown')}",
            f"Benchmark Correlation: {eval.features.get('btc_correlation', 0):.3f}",
            f"Aggression: {self.aggression_level:.2f} | Threshold(after scaling): {float(eval.prediction.get('adaptive_threshold', 0.0)):.3f}",
            "",
            "Evaluation Factors:",
        ]
        for reason in eval.reasons:
            lines.append(f"  • {reason}")
        return "\n".join(lines)

    @staticmethod
    def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
        if abs(float(denominator)) < 1e-12:
            return float(default)
        return float(numerator) / float(denominator)

    @staticmethod
    def _bounded(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.5) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(numeric):
            return float(default)
        return float(max(low, min(high, numeric)))

    def _compute_multifactor_components(
        self,
        pair: str,
        latest_15m_row: pd.Series,
        df_15m: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compute short-horizon multi-factor signal components.
        Uses 1m/5m/15m OHLCV and order-book depth imbalance.
        """
        now = datetime.utcnow()
        try:
            df_1m = self.fetcher.fetch_klines(
                pair,
                "1min",
                start_dt=now - timedelta(hours=6),
            )
        except Exception:
            df_1m = pd.DataFrame()

        try:
            df_5m = self.fetcher.fetch_klines(
                pair,
                "5min",
                start_dt=now - timedelta(hours=24),
            )
        except Exception:
            df_5m = pd.DataFrame()

        base_df_1m = df_1m if not df_1m.empty else df_15m
        base_df_5m = df_5m if not df_5m.empty else df_15m

        c1 = pd.to_numeric(base_df_1m.get("close", pd.Series(dtype=float)), errors="coerce")
        c5 = pd.to_numeric(base_df_5m.get("close", pd.Series(dtype=float)), errors="coerce")
        v5 = pd.to_numeric(base_df_5m.get("volume", pd.Series(dtype=float)), errors="coerce")

        recent_returns = c1.pct_change().dropna().tail(8)
        momentum_raw = float(recent_returns.sum()) if not recent_returns.empty else 0.0
        momentum_factor = self._bounded(0.5 + momentum_raw * 25.0)

        vol_est = float(recent_returns.std()) if len(recent_returns) >= 2 else 0.0
        target_vol = 0.0025
        volatility_factor = self._bounded(1.0 - abs(vol_est - target_vol) / max(target_vol, 1e-9))

        if len(c5.dropna()) >= 25:
            ema_fast = float(c5.ewm(span=9, adjust=False).mean().iloc[-1])
            ema_slow = float(c5.ewm(span=21, adjust=False).mean().iloc[-1])
            trend_delta = self._safe_div(ema_fast - ema_slow, max(abs(ema_slow), 1e-12))
            trend_factor = self._bounded(0.5 + trend_delta * 40.0)
        else:
            trend_factor = self._bounded(float(latest_15m_row.get("regime_score", 0.5)))

        if len(c1.dropna()) >= 20:
            rolling_mean = c1.rolling(20).mean().iloc[-1]
            rolling_std = c1.rolling(20).std().iloc[-1]
            z_score = self._safe_div(float(c1.iloc[-1] - rolling_mean), float(rolling_std), default=0.0)
            mean_reversion_factor = self._bounded(0.5 - z_score * 0.15)
        else:
            mean_reversion_factor = 0.5

        if len(v5.dropna()) >= 20:
            vol_ratio = self._safe_div(float(v5.iloc[-1]), float(v5.tail(20).mean()), default=1.0)
            volume_factor = self._bounded(0.5 + (vol_ratio - 1.0) * 0.30)
        else:
            volume_factor = self._bounded(float(latest_15m_row.get("volume_ratio", 1.0)) * 0.5)

        imbalance = 0.0
        try:
            orderbook = self.fetcher.get_orderbook(pair, depth=20)
            bid_volume = float(sum(float(level[1]) for level in orderbook.get("bids", []) if len(level) >= 2))
            ask_volume = float(sum(float(level[1]) for level in orderbook.get("asks", []) if len(level) >= 2))
            imbalance = self._safe_div(bid_volume - ask_volume, bid_volume + ask_volume, default=0.0)
            imbalance = max(-1.0, min(1.0, float(imbalance)))
        except Exception:
            imbalance = 0.0

        weighted = (
            momentum_factor * self._factor_weights["momentum"]
            + volatility_factor * self._factor_weights["volatility"]
            + trend_factor * self._factor_weights["trend"]
            + mean_reversion_factor * self._factor_weights["mean_reversion"]
            + volume_factor * self._factor_weights["volume"]
        )
        signal_score = self._bounded(weighted + imbalance * 0.10)
        return {
            "signal_score": float(signal_score),
            "momentum_factor": float(momentum_factor),
            "volatility_factor": float(volatility_factor),
            "trend_factor": float(trend_factor),
            "mean_reversion_factor": float(mean_reversion_factor),
            "volume_factor": float(volume_factor),
            "volume_imbalance": float(imbalance),
        }

    def _record_signal_record(
        self,
        *,
        pair: str,
        signal_score: float,
        decision: str,
        side: str,
        prediction: Dict[str, Any],
        reason: str,
    ) -> None:
        session = get_session()
        try:
            model_conf = prediction.get("meta_probability", prediction.get("confidence", 0.0))
            record = SignalRecord(
                timestamp=datetime.utcnow(),
                pair=pair,
                timeframe="1min",
                decision=str(decision).upper(),
                side=str(side).upper(),
                signal_score=self._bounded(signal_score),
                momentum_factor=self._bounded(prediction.get("momentum_factor", 0.5)),
                volatility_factor=self._bounded(prediction.get("volatility_factor", 0.5)),
                trend_factor=self._bounded(prediction.get("trend_factor", 0.5)),
                mean_reversion_factor=self._bounded(prediction.get("mean_reversion_factor", 0.5)),
                volume_factor=self._bounded(prediction.get("volume_factor", 0.5)),
                volume_imbalance=max(-1.0, min(1.0, float(prediction.get("volume_imbalance", 0.0) or 0.0))),
                model_confidence=self._bounded(model_conf),
                regime=str(prediction.get("regime_info", {}).get("trend", "unknown")),
                payload={
                    "reason": reason,
                    "prob_up": float(prediction.get("prob_up", 0.0) or 0.0),
                    "prob_down": float(prediction.get("prob_down", 0.0) or 0.0),
                    "adaptive_threshold": float(prediction.get("adaptive_threshold", 0.0) or 0.0),
                },
            )
            session.add(record)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.debug(f"Signal record persistence skipped for {pair}: {exc}")
        finally:
            session.close()

    def _get_performance_metrics(self, lookback: int = 100) -> Dict[str, float]:
        """
        Pull rolling performance metrics for adaptive threshold control.
        """
        default = {"win_rate": 0.5, "sharpe": 0.0, "drawdown": 0.0}
        session = get_session()
        try:
            latest = (
                session.query(PerformanceMetric)
                .order_by(PerformanceMetric.id.desc())
                .first()
            )
            if latest is not None:
                return {
                    "win_rate": max(0.0, min(1.0, float(latest.win_rate or 0.5))),
                    "sharpe": float(latest.sharpe or 0.0),
                    "drawdown": abs(float(latest.max_drawdown or 0.0)),
                }

            recent_closed = (
                session.query(Trade)
                .filter(Trade.status == "closed", Trade.pnl.isnot(None))
                .order_by(Trade.exit_time.desc(), Trade.id.desc())
                .limit(max(10, int(lookback)))
                .all()
            )
            if not recent_closed:
                return default

            trades = list(reversed(recent_closed))
            pnls = [float(t.pnl or 0.0) for t in trades]
            returns = [float(t.pnl_pct or 0.0) for t in trades]
            wins = sum(1 for pnl in pnls if pnl > 0)
            win_rate = wins / len(pnls) if pnls else 0.5

            mean_return = float(np.mean(returns)) if returns else 0.0
            std_return = self._std(returns)
            sharpe = (mean_return / std_return) if std_return > 1e-12 else 0.0
            drawdown = self._max_drawdown_from_pnl(pnls)

            return {
                "win_rate": max(0.0, min(1.0, win_rate)),
                "sharpe": float(sharpe),
                "drawdown": float(drawdown),
            }
        except Exception as exc:
            logger.debug(f"Adaptive threshold metrics fallback: {exc}")
            return default
        finally:
            session.close()

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean_v = sum(values) / len(values)
        variance = sum((v - mean_v) ** 2 for v in values) / len(values)
        return float(np.sqrt(max(variance, 0.0)))

    @staticmethod
    def _max_drawdown_from_pnl(pnls: List[float]) -> float:
        if not pnls:
            return 0.0
        equity = max(1.0, float(settings.trading.paper_starting_balance))
        peak = equity
        max_dd = 0.0
        for pnl in pnls:
            equity += float(pnl)
            if equity > peak:
                peak = equity
            drawdown = ((peak - equity) / peak) if peak > 0 else 0.0
            if drawdown > max_dd:
                max_dd = drawdown
        return float(max_dd)

    # ------------------------------------------------------------------
    # Dynamic threshold adjustments (called by self-improving loop)
    # ------------------------------------------------------------------

    def update_thresholds(self, adjustments: Dict[str, float]) -> None:
        """Update decision thresholds based on feedback analysis."""
        if "confidence_threshold" in adjustments:
            old = self.confidence_threshold
            self.confidence_threshold = max(0.40, min(0.95, adjustments["confidence_threshold"]))
            if "min_confidence" not in adjustments:
                self.min_confidence = self.confidence_threshold
            logger.info(f"Confidence threshold: {old:.3f} → {self.confidence_threshold:.3f}")

        if "spread_max_pct" in adjustments:
            self._spread_max_pct = adjustments["spread_max_pct"]

        if "min_volume_ratio" in adjustments:
            self._min_volume_ratio = adjustments["min_volume_ratio"]

        if "risk_per_trade" in adjustments:
            self.risk_per_trade = max(0.002, min(0.05, adjustments["risk_per_trade"]))

        if "min_confidence" in adjustments:
            self.min_confidence = max(0.40, min(0.99, adjustments["min_confidence"]))
        self.adaptive_threshold.set_base_threshold(self.min_confidence)

        if "tft_weight" in adjustments:
            self.tft_weight = float(adjustments["tft_weight"])
        if "xgb_weight" in adjustments:
            self.xgb_weight = float(adjustments["xgb_weight"])
        if "ppo_weight" in adjustments:
            self.ppo_weight = float(adjustments["ppo_weight"])
        self._normalize_agent_weights()

    def get_current_thresholds(self) -> Dict[str, Any]:
        """Return current decision thresholds."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "min_confidence": self.min_confidence,
            "adaptive_confidence_threshold": self.adaptive_threshold.last_threshold,
            "aggression_level": self.aggression_level,
            "allow_shorts": self.allow_shorts,
            "risk_per_trade": self.risk_per_trade,
            "spread_max_pct": self._spread_max_pct,
            "min_volume_ratio": self._min_volume_ratio,
            "btc_corr_weight": self._btc_corr_weight,
            "tft_weight": self.tft_weight,
            "xgb_weight": self.xgb_weight,
            "ppo_weight": self.ppo_weight,
        }

    def _normalize_agent_weights(self) -> None:
        self.tft_weight = max(0.05, float(self.tft_weight))
        self.xgb_weight = max(0.05, float(self.xgb_weight))
        self.ppo_weight = max(0.05, float(self.ppo_weight))
        total = self.tft_weight + self.xgb_weight + self.ppo_weight
        if total <= 0:
            self.tft_weight, self.xgb_weight, self.ppo_weight = 0.4, 0.4, 0.2
            return
        self.tft_weight /= total
        self.xgb_weight /= total
        self.ppo_weight /= total

    def _record_no_trade_event(
        self,
        evaluations: List[PairEvaluation],
        valid: List[PairEvaluation],
        reason: str,
    ) -> None:
        """Persist a decision cycle event even when no trade is opened."""
        def _finite_or_none(value: Any, digits: Optional[int] = None) -> Optional[float]:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(numeric):
                return None
            return round(numeric, digits) if digits is not None else numeric

        best_eval = valid[0] if valid else (evaluations[0] if evaluations else None)
        top_candidates = []
        source = valid if valid else evaluations
        for ev in source[:5]:
            top_candidates.append({
                "pair": ev.pair,
                "score": _finite_or_none(ev.score, 4),
                "prob_up": _finite_or_none(ev.prediction.get("prob_up", 0), 4),
                "prob_down": _finite_or_none(ev.prediction.get("prob_down", 0), 4),
                "confidence": _finite_or_none(ev.prediction.get("confidence", 0), 4),
                "disqualified": ev.disqualified,
                "disqualify_reason": ev.disqualify_reason,
            })

        session = get_session()
        try:
            event = DecisionEvent(
                timestamp=datetime.utcnow(),
                mode=settings.trading.trading_mode.upper(),
                status="no_trade",
                reason=reason,
                candidates_evaluated=len(evaluations),
                candidates_valid=len(valid),
                best_pair=best_eval.pair if best_eval else None,
                best_score=_finite_or_none(best_eval.score) if best_eval else None,
                best_ai_score=_finite_or_none(best_eval.prediction.get("meta_probability", 0)) if best_eval else None,
                best_confidence=_finite_or_none(best_eval.prediction.get("confidence", 0)) if best_eval else None,
                best_prob_up=_finite_or_none(best_eval.prediction.get("prob_up", 0)) if best_eval else None,
                best_prob_down=_finite_or_none(best_eval.prediction.get("prob_down", 0)) if best_eval else None,
                regime=best_eval.features.get("market_regime") if best_eval else None,
                volatility_regime=best_eval.features.get("volatility_regime") if best_eval else None,
                adaptive_threshold=_finite_or_none(best_eval.prediction.get("adaptive_threshold")) if best_eval else None,
                top_candidates_json=top_candidates,
            )
            session.add(event)
            session.commit()
            logger.debug(f"Persisted NO_TRADE decision event: reason={reason}")
        except Exception as e:
            session.rollback()
            logger.warning(f"Could not persist decision event: {e}")
        finally:
            session.close()

    def _record_prediction(
        self,
        pair: str,
        prediction: Dict[str, Any],
        features: Dict[str, Any],
    ) -> None:
        if not bool(prediction.get("valid", True)):
            logger.info(f"Skipping prediction persistence for {pair}: invalid forecast payload")
            return

        expected_move = float(prediction.get("expected_move", 0.0))
        if not np.isfinite(expected_move):
            expected_move = 0.0

        session = get_session()
        try:
            confidence_value = float(
                prediction.get("meta_probability", prediction.get("confidence", 0.0))
            )
            if not np.isfinite(confidence_value):
                confidence_value = 0.0
            record = Prediction(
                timestamp=datetime.utcnow(),
                pair=pair,
                timeframe="15min",
                prob_up=float(prediction.get("prob_up", 0.5)),
                prob_down=float(prediction.get("prob_down", 0.5)),
                expected_move=expected_move,
                # Confidence column stores live AI score used for trading decisions.
                confidence=confidence_value,
                volatility_regime=str(features.get("volatility_regime", "unknown")),
                market_regime=str(features.get("market_regime", "unknown")),
                forecast_vector=prediction.get("forecast_vector", []),
                model_version=(
                    f"tft={prediction.get('model_version', '')};"
                    f"xgb={prediction.get('meta_model_version', '')}"
                ),
                acted_on=False,
            )
            session.add(record)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.warning(f"Could not persist prediction for {pair}: {e}")
        finally:
            session.close()
