"""
Unified AI score formula across supervised, RL, Opus, and performance agents.
"""
from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class CombinedAgentInputs:
    supervised_win_probability: float
    supervised_confidence: float
    rl_action: int | None = None
    opus_sentiment: float | None = None  # [-1, 1]
    opus_risk: float | None = None  # [0, 1]
    recent_win_rate: float | None = None  # [0, 1]
    recent_sharpe: float | None = None  # approximately [-1, 2]


def compute_combined_ai_score(inputs: CombinedAgentInputs) -> dict[str, float | str | dict[str, float]]:
    win_component = _clamp(inputs.supervised_win_probability)
    confidence_component = _clamp(inputs.supervised_confidence)

    rl_map = {
        0: 1.0,   # increase exposure
        1: 0.35,  # decrease exposure
        2: 0.65,  # hold
        3: 0.0,   # close / avoid
    }
    rl_component = rl_map.get(int(inputs.rl_action), 0.5) if inputs.rl_action is not None else 0.5

    if inputs.opus_sentiment is None or inputs.opus_risk is None:
        opus_component = 0.5
    else:
        normalized_sentiment = _clamp((float(inputs.opus_sentiment) + 1.0) / 2.0)
        normalized_risk = _clamp(float(inputs.opus_risk))
        opus_component = _clamp(normalized_sentiment * (1.0 - normalized_risk))

    if inputs.recent_win_rate is None and inputs.recent_sharpe is None:
        performance_component = 0.5
    else:
        win_rate_component = _clamp(inputs.recent_win_rate or 0.5)
        sharpe = float(inputs.recent_sharpe or 0.0)
        sharpe_component = _clamp((sharpe + 1.0) / 3.0)
        performance_component = _clamp(0.7 * win_rate_component + 0.3 * sharpe_component)

    weights = {
        "supervised_win_probability": 0.35,
        "supervised_confidence": 0.15,
        "rl_policy": 0.20,
        "opus_reasoning": 0.15,
        "recent_performance": 0.15,
    }

    final_score = (
        weights["supervised_win_probability"] * win_component
        + weights["supervised_confidence"] * confidence_component
        + weights["rl_policy"] * rl_component
        + weights["opus_reasoning"] * opus_component
        + weights["recent_performance"] * performance_component
    )
    final_score = _clamp(final_score)
    return {
        "score": final_score,
        "formula": (
            "0.35*supervised_win + 0.15*supervised_confidence + 0.20*rl_policy + "
            "0.15*opus_reasoning + 0.15*recent_performance"
        ),
        "components": {
            "supervised_win": win_component,
            "supervised_confidence": confidence_component,
            "rl_policy": rl_component,
            "opus_reasoning": opus_component,
            "recent_performance": performance_component,
        },
    }
