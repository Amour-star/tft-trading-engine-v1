"""
Opus integration and combined AI score endpoints.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import desc

from tft_engine.ai.scoring import CombinedAgentInputs, compute_combined_ai_score
from tft_engine.api.runtime import get_runtime
from tft_engine.database.connection import get_session
from tft_engine.database.models import AIMetric

router = APIRouter(prefix="/api/ai", tags=["ai"])


class OpusAnalyzeRequest(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"] = "BUY"
    market_snapshot: dict[str, Any] = Field(default_factory=dict)
    user_prompt: Optional[str] = None
    max_output_tokens: int = Field(default=400, ge=1, le=800)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    use_cache: bool = True


class CombinedScoreRequest(BaseModel):
    supervised_win_probability: float = Field(ge=0.0, le=1.0)
    supervised_confidence: float = Field(ge=0.0, le=1.0)
    rl_action: Optional[int] = Field(default=None, ge=0, le=3)
    opus_sentiment: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    opus_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recent_win_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recent_sharpe: Optional[float] = None


class LiveScoreRequest(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"] = "BUY"
    entry_price: float = Field(gt=0)
    current_price: Optional[float] = None
    rsi: float = 50.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    atr: float = 0.0
    volatility: float = 0.0
    volume: float = 0.0
    macd: float = 0.0
    market_regime: str = "sideways"
    include_opus: bool = False
    opus_note: Optional[str] = None
    opus_max_output_tokens: int = Field(default=300, ge=1, le=800)


def _latest_performance() -> tuple[float, float]:
    session = get_session()
    try:
        metric = session.query(AIMetric).order_by(desc(AIMetric.date)).first()
        if not metric:
            return 0.5, 0.0
        return float(metric.win_rate), float(metric.sharpe_ratio)
    finally:
        session.close()


@router.post("/opus/analyze")
def opus_analyze(payload: OpusAnalyzeRequest):
    runtime = get_runtime()
    opus_service = runtime["opus_service"]
    try:
        result = opus_service.analyze(
            symbol=payload.symbol,
            side=payload.side,
            market_snapshot=payload.market_snapshot,
            user_prompt=payload.user_prompt,
            max_output_tokens=payload.max_output_tokens,
            temperature=payload.temperature,
            use_cache=payload.use_cache,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Opus analyze failed: {exc}") from exc

    return {
        "model": result.model,
        "recommendation": result.recommendation,
        "rationale": result.rationale,
        "sentiment_score": result.sentiment_score,
        "risk_score": result.risk_score,
        "confidence": result.confidence,
        "raw_text": result.raw_text,
        "usage": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "from_cache": result.from_cache,
        },
        "budget": result.budget,
    }


@router.get("/opus/usage")
def opus_usage(days: int = 7):
    runtime = get_runtime()
    opus_service = runtime["opus_service"]
    return opus_service.usage_summary(days=max(1, min(int(days), 90)))


@router.post("/score/combined")
def combined_score(payload: CombinedScoreRequest):
    score = compute_combined_ai_score(
        CombinedAgentInputs(
            supervised_win_probability=payload.supervised_win_probability,
            supervised_confidence=payload.supervised_confidence,
            rl_action=payload.rl_action,
            opus_sentiment=payload.opus_sentiment,
            opus_risk=payload.opus_risk,
            recent_win_rate=payload.recent_win_rate,
            recent_sharpe=payload.recent_sharpe,
        )
    )
    return score


@router.post("/score/live")
def live_score(payload: LiveScoreRequest):
    runtime = get_runtime()
    controller = runtime["ai_controller"]
    opus_service = runtime["opus_service"]

    snapshot = payload.model_dump()
    if snapshot.get("current_price") is None:
        snapshot["current_price"] = snapshot["entry_price"]

    try:
        supervised_pred = controller.supervised.predict(snapshot)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Supervised model unavailable: {exc}") from exc

    rl_action: Optional[int] = None
    if controller.rl is not None:
        try:
            obs = controller._rl_observation(snapshot, quantity=1.0)  # noqa: SLF001
            rl_action = int(controller.rl.choose_action(obs))
        except Exception:
            rl_action = None

    opus_payload = None
    opus_sentiment = None
    opus_risk = None
    if payload.include_opus:
        try:
            opus_result = opus_service.analyze(
                symbol=payload.symbol,
                side=payload.side,
                market_snapshot=snapshot,
                user_prompt=payload.opus_note,
                max_output_tokens=payload.opus_max_output_tokens,
                temperature=0.2,
                use_cache=True,
            )
            opus_payload = {
                "recommendation": opus_result.recommendation,
                "rationale": opus_result.rationale,
                "confidence": opus_result.confidence,
                "cost_usd": opus_result.cost_usd,
                "input_tokens": opus_result.input_tokens,
                "output_tokens": opus_result.output_tokens,
                "from_cache": opus_result.from_cache,
                "budget": opus_result.budget,
            }
            opus_sentiment = opus_result.sentiment_score
            opus_risk = opus_result.risk_score
        except Exception as exc:
            opus_payload = {"error": str(exc)}

    recent_win_rate, recent_sharpe = _latest_performance()
    combined = compute_combined_ai_score(
        CombinedAgentInputs(
            supervised_win_probability=float(supervised_pred.win_probability),
            supervised_confidence=float(supervised_pred.confidence_score),
            rl_action=rl_action,
            opus_sentiment=opus_sentiment,
            opus_risk=opus_risk,
            recent_win_rate=recent_win_rate,
            recent_sharpe=recent_sharpe,
        )
    )
    return {
        "score": combined["score"],
        "formula": combined["formula"],
        "components": combined["components"],
        "signals": {
            "supervised": {
                "win_probability": float(supervised_pred.win_probability),
                "confidence_score": float(supervised_pred.confidence_score),
                "expected_return": float(supervised_pred.expected_return),
                "model_version": supervised_pred.model_version,
            },
            "rl": {"action": rl_action},
            "performance": {
                "recent_win_rate": recent_win_rate,
                "recent_sharpe": recent_sharpe,
            },
            "opus": opus_payload,
        },
    }
