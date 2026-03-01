"""
Anthropic Opus integration service with caching and cost controls.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import desc, func

from tft_engine.ai.llm.cost_control import LLMCostController
from tft_engine.config import config
from tft_engine.database.models import LLMUsage


def _clamp(v: float, low: float, high: float) -> float:
    return max(low, min(high, float(v)))


def estimate_tokens(text: str) -> int:
    # Practical approximation for budget checks before provider usage metrics are known.
    return max(1, int(len(text) / 4))


def _extract_text(content_blocks: Any) -> str:
    if not isinstance(content_blocks, list):
        return ""
    parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts).strip()


def _try_parse_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


@dataclass(frozen=True)
class OpusResult:
    model: str
    raw_text: str
    recommendation: str
    rationale: str
    sentiment_score: float
    risk_score: float
    confidence: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    from_cache: bool
    budget: dict[str, float | str | bool]
    latency_ms: float


class OpusIntegrationService:
    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory
        self._cost = LLMCostController(session_factory)
        self._cache: dict[str, tuple[float, OpusResult]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _build_prompt(
        symbol: str,
        side: str,
        market_snapshot: dict[str, Any],
        user_prompt: str | None,
    ) -> str:
        snapshot_json = json.dumps(market_snapshot, ensure_ascii=True, sort_keys=True)
        extra = user_prompt.strip() if user_prompt else "No additional trader note."
        return (
            "You are an institutional crypto risk analyst for an automated trading engine.\n"
            "Return JSON only with fields: recommendation, rationale, sentiment_score, risk_score, confidence, risk_flags.\n"
            "Rules: recommendation is one of BUY, SELL, HOLD. sentiment_score in [-1,1]. "
            "risk_score in [0,1] where 1 is highest risk. confidence in [0,1].\n"
            f"Trade side hint: {side}\n"
            f"Symbol: {symbol}\n"
            f"Market snapshot JSON: {snapshot_json}\n"
            f"Trader note: {extra}\n"
        )

    @staticmethod
    def _cache_key(model: str, prompt: str, max_output_tokens: int, temperature: float) -> str:
        payload = f"{model}|{max_output_tokens}|{temperature:.3f}|{prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _record_usage(
        self,
        *,
        status: str,
        reason: str | None,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        actual_cost_usd: float,
        cached: bool,
        latency_ms: float | None = None,
    ) -> None:
        session = self._session_factory()
        try:
            session.add(
                LLMUsage(
                    created_at=datetime.utcnow(),
                    provider="anthropic",
                    model=config.opus_model,
                    endpoint="/api/ai/opus/analyze",
                    input_tokens=int(max(0, input_tokens)),
                    output_tokens=int(max(0, output_tokens)),
                    estimated_cost_usd=float(max(0.0, estimated_cost_usd)),
                    actual_cost_usd=float(max(0.0, actual_cost_usd)),
                    cached=1 if cached else 0,
                    status=status,
                    reason=reason,
                    latency_ms=latency_ms,
                )
            )
            session.commit()
        finally:
            session.close()

    def analyze(
        self,
        *,
        symbol: str,
        side: str,
        market_snapshot: dict[str, Any],
        user_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float = 0.2,
        use_cache: bool = True,
    ) -> OpusResult:
        if not config.opus_enabled:
            raise RuntimeError("Opus integration is disabled. Set OPUS_ENABLED=true to enable.")
        if not config.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

        max_out = int(max_output_tokens or config.llm_max_output_tokens)
        max_out = max(1, min(max_out, int(config.llm_max_output_tokens)))
        temperature = _clamp(temperature, 0.0, 1.0)

        prompt = self._build_prompt(
            symbol=symbol,
            side=side,
            market_snapshot=market_snapshot,
            user_prompt=user_prompt,
        )
        estimated_input = min(estimate_tokens(prompt), int(config.llm_max_input_tokens))
        decision = self._cost.decide(input_tokens=estimated_input, output_tokens=max_out)
        budget = {
            "allowed": decision.allowed,
            "reason": decision.reason,
            "estimated_cost_usd": decision.estimated_cost_usd,
            "day_spend_usd": decision.day_spend_usd,
            "month_spend_usd": decision.month_spend_usd,
            "day_remaining_usd": decision.day_remaining_usd,
            "month_remaining_usd": decision.month_remaining_usd,
        }
        if not decision.allowed:
            self._record_usage(
                status="blocked",
                reason=decision.reason,
                input_tokens=estimated_input,
                output_tokens=max_out,
                estimated_cost_usd=decision.estimated_cost_usd,
                actual_cost_usd=0.0,
                cached=False,
            )
            raise RuntimeError(decision.reason)

        key = self._cache_key(config.opus_model, prompt, max_out, temperature)
        if use_cache:
            now = time.time()
            with self._lock:
                row = self._cache.get(key)
                if row and row[0] > now:
                    cached = row[1]
                    self._record_usage(
                        status="success",
                        reason="cache_hit",
                        input_tokens=0,
                        output_tokens=0,
                        estimated_cost_usd=0.0,
                        actual_cost_usd=0.0,
                        cached=True,
                        latency_ms=0.0,
                    )
                    return OpusResult(
                        model=cached.model,
                        raw_text=cached.raw_text,
                        recommendation=cached.recommendation,
                        rationale=cached.rationale,
                        sentiment_score=cached.sentiment_score,
                        risk_score=cached.risk_score,
                        confidence=cached.confidence,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        from_cache=True,
                        budget=budget,
                        latency_ms=0.0,
                    )

        req_payload = {
            "model": config.opus_model,
            "max_tokens": max_out,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=float(config.llm_timeout_seconds)) as client:
                response = client.post(
                    config.anthropic_base_url,
                    headers={
                        "x-api-key": config.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=req_payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            self._record_usage(
                status="error",
                reason=f"{type(exc).__name__}: {exc}",
                input_tokens=estimated_input,
                output_tokens=max_out,
                estimated_cost_usd=decision.estimated_cost_usd,
                actual_cost_usd=0.0,
                cached=False,
            )
            raise RuntimeError(f"Opus request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        raw_text = _extract_text(data.get("content", []))
        parsed = _try_parse_json(raw_text)

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", estimated_input))
        output_tokens = int(usage.get("output_tokens", max_out))
        actual_cost = self._cost.estimate_cost_usd(input_tokens, output_tokens)

        result = OpusResult(
            model=str(data.get("model") or config.opus_model),
            raw_text=raw_text,
            recommendation=str(parsed.get("recommendation", "HOLD")).upper(),
            rationale=str(parsed.get("rationale", raw_text[:1000])),
            sentiment_score=float(_clamp(parsed.get("sentiment_score", 0.0), -1.0, 1.0)),
            risk_score=float(_clamp(parsed.get("risk_score", 0.5), 0.0, 1.0)),
            confidence=float(_clamp(parsed.get("confidence", 0.5), 0.0, 1.0)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=actual_cost,
            from_cache=False,
            budget=budget,
            latency_ms=latency_ms,
        )

        self._record_usage(
            status="success",
            reason=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=decision.estimated_cost_usd,
            actual_cost_usd=actual_cost,
            cached=False,
            latency_ms=latency_ms,
        )

        if use_cache:
            with self._lock:
                self._cache[key] = (time.time() + int(config.llm_cache_ttl_seconds), result)
        return result

    def usage_summary(self, days: int = 7) -> dict[str, Any]:
        # Compatibility helper: avoid depending on numpy/pandas for summarization.
        session = self._session_factory()
        try:
            cutoff = datetime.utcnow() - timedelta(days=max(1, int(days)))
            rows = (
                session.query(LLMUsage)
                .filter(LLMUsage.created_at >= cutoff)
                .order_by(desc(LLMUsage.created_at))
                .all()
            )
            total_cost = float(sum(r.actual_cost_usd or 0.0 for r in rows))
            total_input_tokens = int(sum(r.input_tokens or 0 for r in rows))
            total_output_tokens = int(sum(r.output_tokens or 0 for r in rows))
            blocked = int(sum(1 for r in rows if r.status == "blocked"))
            errors = int(sum(1 for r in rows if r.status == "error"))
            cached = int(sum(1 for r in rows if r.cached == 1))
            by_status = {}
            for r in rows:
                by_status[r.status] = int(by_status.get(r.status, 0) + 1)

            today = datetime.utcnow()
            day_start = datetime(today.year, today.month, today.day)
            day_spend = (
                session.query(func.sum(LLMUsage.actual_cost_usd))
                .filter(LLMUsage.created_at >= day_start, LLMUsage.status == "success")
                .scalar()
                or 0.0
            )
            month_start = datetime(today.year, today.month, 1)
            month_spend = (
                session.query(func.sum(LLMUsage.actual_cost_usd))
                .filter(LLMUsage.created_at >= month_start, LLMUsage.status == "success")
                .scalar()
                or 0.0
            )
            return {
                "window_days": int(days),
                "request_count": len(rows),
                "total_cost_usd": float(total_cost),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "blocked_requests": blocked,
                "error_requests": errors,
                "cache_hits": cached,
                "status_counts": by_status,
                "day_spend_usd": float(day_spend),
                "month_spend_usd": float(month_spend),
                "day_budget_usd": float(config.llm_daily_budget_usd),
                "month_budget_usd": float(config.llm_monthly_budget_usd),
                "day_remaining_usd": max(0.0, float(config.llm_daily_budget_usd) - float(day_spend)),
                "month_remaining_usd": max(0.0, float(config.llm_monthly_budget_usd) - float(month_spend)),
            }
        finally:
            session.close()
