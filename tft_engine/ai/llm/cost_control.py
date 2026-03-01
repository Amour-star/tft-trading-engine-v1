"""
Cost estimation and budget enforcement for LLM usage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func

from tft_engine.config import config
from tft_engine.database.models import LLMUsage


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str
    estimated_cost_usd: float
    day_spend_usd: float
    month_spend_usd: float
    day_remaining_usd: float
    month_remaining_usd: float


class LLMCostController:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    @staticmethod
    def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
        in_cost = (max(0, int(input_tokens)) / 1000.0) * float(config.llm_cost_input_per_1k_tokens)
        out_cost = (max(0, int(output_tokens)) / 1000.0) * float(config.llm_cost_output_per_1k_tokens)
        return float(in_cost + out_cost)

    def _spend_between(self, start: datetime, end: datetime) -> float:
        session = self._session_factory()
        try:
            total = (
                session.query(func.sum(LLMUsage.actual_cost_usd))
                .filter(LLMUsage.created_at >= start, LLMUsage.created_at < end, LLMUsage.status == "success")
                .scalar()
            )
            return float(total or 0.0)
        finally:
            session.close()

    def current_spend(self) -> tuple[float, float]:
        now = datetime.utcnow()
        day_start = datetime(now.year, now.month, now.day)
        next_day = day_start + timedelta(days=1)
        month_start = datetime(now.year, now.month, 1)
        if now.month == 12:
            next_month = datetime(now.year + 1, 1, 1)
        else:
            next_month = datetime(now.year, now.month + 1, 1)
        day_spend = self._spend_between(day_start, next_day)
        month_spend = self._spend_between(month_start, next_month)
        return day_spend, month_spend

    def decide(self, input_tokens: int, output_tokens: int) -> BudgetDecision:
        estimated = self.estimate_cost_usd(input_tokens=input_tokens, output_tokens=output_tokens)
        day_spend, month_spend = self.current_spend()

        day_remaining = max(0.0, float(config.llm_daily_budget_usd) - day_spend)
        month_remaining = max(0.0, float(config.llm_monthly_budget_usd) - month_spend)

        if estimated > float(config.llm_max_request_cost_usd):
            return BudgetDecision(
                allowed=False,
                reason="Estimated request cost exceeds per-request cap.",
                estimated_cost_usd=estimated,
                day_spend_usd=day_spend,
                month_spend_usd=month_spend,
                day_remaining_usd=day_remaining,
                month_remaining_usd=month_remaining,
            )

        if estimated > day_remaining:
            return BudgetDecision(
                allowed=False,
                reason="Daily LLM budget exhausted.",
                estimated_cost_usd=estimated,
                day_spend_usd=day_spend,
                month_spend_usd=month_spend,
                day_remaining_usd=day_remaining,
                month_remaining_usd=month_remaining,
            )

        if estimated > month_remaining:
            return BudgetDecision(
                allowed=False,
                reason="Monthly LLM budget exhausted.",
                estimated_cost_usd=estimated,
                day_spend_usd=day_spend,
                month_spend_usd=month_spend,
                day_remaining_usd=day_remaining,
                month_remaining_usd=month_remaining,
            )

        return BudgetDecision(
            allowed=True,
            reason="Allowed",
            estimated_cost_usd=estimated,
            day_spend_usd=day_spend,
            month_spend_usd=month_spend,
            day_remaining_usd=day_remaining,
            month_remaining_usd=month_remaining,
        )
