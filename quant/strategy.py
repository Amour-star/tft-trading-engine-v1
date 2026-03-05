from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from quant.config import QuantEngineConfig
from quant.types import FeaturePacket, MarketSnapshot, RegimeState, StrategySignal


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


@dataclass
class StrategyCandidate:
    name: str
    params: Dict[str, float]
    score: float
    sharpe: float
    win_rate: float


class AutoStrategyDiscovery:
    """Generates and ranks parameterized short-horizon strategies."""

    def __init__(self, cfg: QuantEngineConfig) -> None:
        self.cfg = cfg
        self.best_params: Dict[str, Dict[str, float]] = {
            "momentum_breakout": {"threshold": 0.4, "take": 0.004, "stop": 0.0025},
            "mean_reversion": {"threshold": 0.6, "take": 0.003, "stop": 0.002},
            "volatility_breakout": {"threshold": 0.5, "take": 0.005, "stop": 0.0035},
            "orderflow_imbalance": {"threshold": 0.3, "take": 0.0035, "stop": 0.0025},
        }
        self.leaderboard: Dict[str, List[StrategyCandidate]] = {}

    def discover(self, symbol: str, frame_1m: pd.DataFrame) -> None:
        if frame_1m.empty or len(frame_1m) < 80:
            return
        close = pd.to_numeric(frame_1m["close"], errors="coerce").ffill()
        returns = close.pct_change().fillna(0.0)
        sampled: Dict[str, List[StrategyCandidate]] = {
            "momentum_breakout": [],
            "mean_reversion": [],
            "volatility_breakout": [],
            "orderflow_imbalance": [],
        }
        for _ in range(self.cfg.strategy_discovery_samples):
            for name in sampled.keys():
                params = {
                    "threshold": random.uniform(0.2, 0.9),
                    "take": random.uniform(0.0015, 0.0080),
                    "stop": random.uniform(0.0010, 0.0060),
                }
                score, sharpe, win_rate = self._quick_backtest(returns, name, params)
                sampled[name].append(
                    StrategyCandidate(
                        name=name,
                        params=params,
                        score=score,
                        sharpe=sharpe,
                        win_rate=win_rate,
                    )
                )

        for name, candidates in sampled.items():
            ranked = sorted(candidates, key=lambda row: row.score, reverse=True)
            self.leaderboard[f"{symbol}:{name}"] = ranked[:10]
            if ranked:
                self.best_params[name] = dict(ranked[0].params)

    def _quick_backtest(
        self,
        returns: pd.Series,
        strategy_name: str,
        params: Dict[str, float],
    ) -> Tuple[float, float, float]:
        threshold = _safe_float(params.get("threshold"), 0.5)
        signal = np.zeros(len(returns))
        x = returns.values

        if strategy_name == "momentum_breakout":
            signal[1:] = np.where(x[:-1] > threshold * np.std(x), 1.0, -1.0)
        elif strategy_name == "mean_reversion":
            z = (x - np.mean(x)) / max(np.std(x), 1e-9)
            signal = np.where(z > threshold, -1.0, np.where(z < -threshold, 1.0, 0.0))
        elif strategy_name == "volatility_breakout":
            rolling = pd.Series(x).rolling(12).std().fillna(0.0).values
            signal = np.where(rolling > threshold * np.mean(rolling + 1e-9), np.sign(x), 0.0)
        else:
            momentum = pd.Series(x).rolling(3).mean().fillna(0.0).values
            signal = np.where(momentum > threshold * np.std(x), 1.0, np.where(momentum < -threshold * np.std(x), -1.0, 0.0))

        strat_ret = signal[:-1] * x[1:]
        if len(strat_ret) == 0:
            return 0.0, 0.0, 0.0
        mean = float(np.mean(strat_ret))
        std = float(np.std(strat_ret))
        sharpe = mean / std * np.sqrt(60.0 * 24.0) if std > 1e-9 else 0.0
        wins = float(np.mean(strat_ret > 0))
        score = mean * 1000.0 + sharpe * 0.25 + wins * 0.50
        return score, sharpe, wins


class StrategyEngine:
    """Aggregates multi-strategy outputs into a single executable signal."""

    def __init__(self, cfg: QuantEngineConfig) -> None:
        self.cfg = cfg
        self.discovery = AutoStrategyDiscovery(cfg)

    def discover_for_symbol(self, symbol: str, snapshot: MarketSnapshot) -> None:
        frame_1m = snapshot.frames.get("1min")
        if frame_1m is None or frame_1m.empty:
            return
        self.discovery.discover(symbol, frame_1m.tail(self.cfg.strategy_discovery_lookback))

    def generate_signal(
        self,
        snapshot: MarketSnapshot,
        packet: FeaturePacket,
        regime: RegimeState,
    ) -> StrategySignal:
        f = packet.normalized_features
        momentum = _safe_float(f.get("momentum_5"), 0.0)
        reversion = -_safe_float(f.get("vwap_deviation"), 0.0)
        volatility = _safe_float(f.get("volatility_30"), 0.0)
        orderflow = _safe_float(f.get("orderbook_imbalance"), 0.0) + _safe_float(
            f.get("volume_imbalance"), 0.0
        )

        s1 = self._momentum_breakout(momentum, volatility)
        s2 = self._mean_reversion(reversion, momentum)
        s3 = self._volatility_breakout(volatility, momentum)
        s4 = self._orderflow_imbalance(orderflow, momentum)

        scores = {
            "momentum_breakout": s1,
            "mean_reversion": s2,
            "volatility_breakout": s3,
            "orderflow_imbalance": s4,
        }
        weighted_score, components = self._weighted_ensemble(snapshot.symbol, scores, regime.label)

        direction = 1 if weighted_score > 0.08 else (-1 if weighted_score < -0.08 else 0)
        confidence = min(0.99, max(0.0, abs(weighted_score)))
        if direction == 0:
            confidence = min(confidence, 0.45)
        best_name = max(scores.items(), key=lambda kv: abs(kv[1]))[0]
        reason = (
            f"ensemble={weighted_score:.3f} "
            f"regime={regime.label} "
            f"spread={snapshot.spread_pct:.5f}"
        )
        return StrategySignal(
            symbol=snapshot.symbol,
            timestamp=datetime.utcnow(),
            direction=direction,
            confidence=confidence,
            score=weighted_score,
            strategy_name=best_name,
            regime=regime.label,
            reason=reason,
            components=components,
        )

    def _weighted_ensemble(
        self,
        symbol: str,
        scores: Dict[str, float],
        regime_label: str,
    ) -> Tuple[float, Dict[str, float]]:
        defaults = {
            "momentum_breakout": 0.30,
            "mean_reversion": 0.24,
            "volatility_breakout": 0.22,
            "orderflow_imbalance": 0.24,
        }
        if regime_label == "Trending":
            defaults["momentum_breakout"] += 0.10
        if regime_label == "Mean Reverting":
            defaults["mean_reversion"] += 0.12
        if regime_label == "High Volatility":
            defaults["volatility_breakout"] += 0.10
        if regime_label == "Low Volatility":
            defaults["orderflow_imbalance"] += 0.07

        # Blend with discovery leaderboard score when available.
        blend = {}
        for name, value in defaults.items():
            top = self.discovery.leaderboard.get(f"{symbol}:{name}", [])
            bonus = 0.0
            if top:
                bonus = min(0.12, max(0.0, _safe_float(top[0].score) / 10.0))
            blend[name] = value + bonus

        total_w = sum(blend.values()) or 1.0
        normalized_w = {name: weight / total_w for name, weight in blend.items()}
        weighted_score = sum(_safe_float(scores[k]) * normalized_w[k] for k in normalized_w)
        return float(weighted_score), {k: float(v) for k, v in normalized_w.items()}

    def _momentum_breakout(self, momentum: float, volatility: float) -> float:
        params = self.discovery.best_params.get("momentum_breakout", {})
        threshold = _safe_float(params.get("threshold"), 0.4) * 0.01
        if momentum > threshold:
            return min(1.0, momentum * 20.0 / max(volatility + 1e-5, 1e-4))
        if momentum < -threshold:
            return max(-1.0, momentum * 20.0 / max(volatility + 1e-5, 1e-4))
        return 0.0

    def _mean_reversion(self, reversion: float, momentum: float) -> float:
        params = self.discovery.best_params.get("mean_reversion", {})
        threshold = _safe_float(params.get("threshold"), 0.6)
        if reversion > threshold * 0.4:
            return 0.8 - momentum * 4.0
        if reversion < -threshold * 0.4:
            return -0.8 - momentum * 4.0
        return -momentum * 1.5

    def _volatility_breakout(self, volatility: float, momentum: float) -> float:
        params = self.discovery.best_params.get("volatility_breakout", {})
        threshold = _safe_float(params.get("threshold"), 0.5)
        if volatility > threshold:
            return np.sign(momentum) * min(1.0, abs(momentum) * 10.0 + volatility * 0.5)
        return np.sign(momentum) * min(0.4, abs(momentum) * 4.0)

    def _orderflow_imbalance(self, orderflow: float, momentum: float) -> float:
        params = self.discovery.best_params.get("orderflow_imbalance", {})
        threshold = _safe_float(params.get("threshold"), 0.3)
        score = orderflow * 1.2 + momentum * 0.8
        if abs(orderflow) < threshold * 0.2:
            score *= 0.6
        return float(max(-1.0, min(1.0, score)))
