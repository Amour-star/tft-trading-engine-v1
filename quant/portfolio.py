from __future__ import annotations

from typing import Dict, List

import numpy as np

from quant.types import FeaturePacket, PortfolioTarget, StrategySignal


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


class PortfolioOptimizer:
    """Dynamic capital allocator targeting Sharpe and drawdown control."""

    def optimize(
        self,
        total_equity: float,
        signals: Dict[str, StrategySignal],
        features: Dict[str, FeaturePacket],
        risk_factor: float,
    ) -> List[PortfolioTarget]:
        if total_equity <= 0:
            return []
        symbols = list(signals.keys())
        if not symbols:
            return []

        raw_weights: Dict[str, float] = {}
        for symbol in symbols:
            signal = signals[symbol]
            packet = features[symbol]
            vol = abs(_safe_float(packet.raw_features.get("realized_volatility"), 0.0))
            vol_adj = 1.0 / max(0.05, vol)
            edge = abs(_safe_float(signal.score)) * max(0.01, signal.confidence)
            raw_weights[symbol] = max(0.0, edge * vol_adj)

        sum_w = sum(raw_weights.values())
        if sum_w <= 1e-9:
            equal_weight = 1.0 / max(1, len(symbols))
            raw_weights = {symbol: equal_weight for symbol in symbols}
            sum_w = 1.0

        targets: List[PortfolioTarget] = []
        for symbol in symbols:
            normalized = raw_weights[symbol] / sum_w
            capped = min(0.35, normalized)
            risk_budget = total_equity * risk_factor * capped
            target_notional = total_equity * capped
            targets.append(
                PortfolioTarget(
                    symbol=symbol,
                    target_weight=capped,
                    target_notional=target_notional,
                    risk_budget=risk_budget,
                )
            )

        # Re-normalize after cap.
        total_w = sum(t.target_weight for t in targets) or 1.0
        for target in targets:
            target.target_weight /= total_w
            target.target_notional = total_equity * target.target_weight
            target.risk_budget = total_equity * risk_factor * target.target_weight
        return targets
