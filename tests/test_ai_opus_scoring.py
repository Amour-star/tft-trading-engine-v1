from __future__ import annotations

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tft_engine.ai.llm.cost_control import LLMCostController
from tft_engine.ai.scoring import CombinedAgentInputs, compute_combined_ai_score
from tft_engine.database.models import Base, LLMUsage


def test_combined_score_is_clamped_and_weighted():
    result = compute_combined_ai_score(
        CombinedAgentInputs(
            supervised_win_probability=0.8,
            supervised_confidence=0.6,
            rl_action=0,
            opus_sentiment=0.5,
            opus_risk=0.2,
            recent_win_rate=0.55,
            recent_sharpe=1.0,
        )
    )
    assert 0.0 <= float(result["score"]) <= 1.0
    components = result["components"]
    assert float(components["supervised_win"]) == 0.8
    assert float(components["rl_policy"]) == 1.0


def test_cost_controller_blocks_expensive_request():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    controller = LLMCostController(SessionLocal)

    decision = controller.decide(input_tokens=5000, output_tokens=20000)
    assert decision.allowed is False
    assert "per-request cap" in decision.reason.lower()


def test_cost_controller_tracks_spend_from_success_rows():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = SessionLocal()
    try:
        session.add(
            LLMUsage(
                created_at=datetime.utcnow(),
                provider="anthropic",
                model="claude-opus-4-1",
                endpoint="/api/ai/opus/analyze",
                input_tokens=1000,
                output_tokens=500,
                estimated_cost_usd=0.05,
                actual_cost_usd=0.07,
                cached=0,
                status="success",
                reason=None,
                latency_ms=120.0,
            )
        )
        session.commit()
    finally:
        session.close()

    controller = LLMCostController(SessionLocal)
    day_spend, month_spend = controller.current_spend()
    assert day_spend >= 0.07
    assert month_spend >= 0.07
