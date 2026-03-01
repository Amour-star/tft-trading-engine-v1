"""
Optional LLM governance layer for pre-trade approval and AI score adjustments.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Tuple

import httpx
from loguru import logger
from sqlalchemy import func

from config.settings import settings
from data.database import AIDecisionAudit, GovernanceLog, LLMUsage, get_session

if TYPE_CHECKING:
    from engine.decision import TradeSignal


FALLBACK_RESPONSE: Dict[str, Any] = {
    "approve": True,
    "size_mult": 1.0,
    "conf_adj": 0.0,
    "risk_mode": "neutral",
    "code": "LLM_FAILSAFE",
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


@dataclass(frozen=True)
class GovernanceDecision:
    approve: bool
    size_mult: float
    conf_adj: float
    risk_mode: str
    code: str
    from_cache: bool = False
    fallback_reason: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    raw_response: Dict[str, Any] | None = None


class GovernanceService:
    """Bounded-latency governance wrapper with deterministic fallback."""

    _RISK_MODES = {"neutral", "defensive", "aggressive"}

    def __init__(self) -> None:
        self._enabled = settings.governance.llm_enabled
        self._cache: Dict[Tuple[str, str, str], tuple[float, GovernanceDecision]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="tft-governance",
        )
        self._failure_count = 0
        self._auto_disabled = False
        self._active_date = datetime.now().date()
        self._daily_spend_usd = self._query_today_spend()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def evaluate(
        self,
        signal: "TradeSignal",
        ppo_size_mult: float,
    ) -> GovernanceDecision:
        self._reset_daily_counters_if_needed()

        if not self._enabled:
            return self._fallback("disabled")

        if self._auto_disabled:
            decision = self._fallback("auto_disabled")
            self._record_governance_log(signal, {}, decision, None)
            return decision

        rsi_bucket = self._to_rsi_bucket(signal.features_snapshot.get("rsi_14", 50.0))
        volatility_regime = str(signal.volatility_regime or "unknown")
        cache_key = (signal.pair, volatility_regime, rsi_bucket)

        cached = self._get_cached(cache_key)
        if cached is not None:
            decision = GovernanceDecision(
                **{**cached.__dict__, "from_cache": True},
            )
            self._record_usage(
                signal=signal,
                status="cache_hit",
                error=None,
                prompt_tokens=0,
                completion_tokens=0,
                estimated_cost_usd=0.0,
                actual_cost_usd=0.0,
                latency_ms=0.0,
            )
            self._record_governance_log(
                signal,
                {"cache_key": list(cache_key)},
                decision,
                None,
            )
            return decision

        request_payload = self._build_request_payload(
            signal=signal,
            ppo_size_mult=ppo_size_mult,
            volatility_regime=volatility_regime,
            rsi_bucket=rsi_bucket,
        )
        prompt = json.dumps(request_payload, separators=(",", ":"), sort_keys=True)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = max(1, int(settings.governance.max_output_tokens))
        estimated_cost = self._estimate_cost(prompt_tokens, completion_tokens)

        if self._is_cost_blocked(estimated_cost):
            decision = self._fallback("cost_exceeded")
            decision = GovernanceDecision(
                **{
                    **decision.__dict__,
                    "estimated_cost_usd": estimated_cost,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
            self._record_usage(
                signal=signal,
                status="blocked",
                error="cost_exceeded",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost,
                actual_cost_usd=0.0,
                latency_ms=0.0,
            )
            self._record_failure()
            self._record_governance_log(signal, request_payload, decision, "cost_exceeded")
            return decision

        start = time.perf_counter()
        future = self._executor.submit(self._request_llm, prompt)
        try:
            response_obj = future.result(timeout=2.0)
        except TimeoutError:
            decision = self._fallback("timeout")
            latency_ms = (time.perf_counter() - start) * 1000.0
            decision = GovernanceDecision(
                **{
                    **decision.__dict__,
                    "latency_ms": latency_ms,
                    "estimated_cost_usd": estimated_cost,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
            self._record_usage(
                signal=signal,
                status="timeout",
                error="timeout",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost,
                actual_cost_usd=0.0,
                latency_ms=latency_ms,
            )
            self._record_failure()
            self._record_governance_log(signal, request_payload, decision, "timeout")
            return decision
        except Exception as exc:
            decision = self._fallback("api_error")
            latency_ms = (time.perf_counter() - start) * 1000.0
            decision = GovernanceDecision(
                **{
                    **decision.__dict__,
                    "latency_ms": latency_ms,
                    "estimated_cost_usd": estimated_cost,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
            self._record_usage(
                signal=signal,
                status="error",
                error=str(exc),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost,
                actual_cost_usd=0.0,
                latency_ms=latency_ms,
            )
            self._record_failure()
            self._record_governance_log(signal, request_payload, decision, "api_error")
            return decision

        latency_ms = float(response_obj.get("latency_ms", (time.perf_counter() - start) * 1000.0))
        raw_text = str(response_obj.get("raw_text", "")).strip()
        usage = response_obj.get("usage") or {}
        completion_tokens = int(usage.get("output_tokens", completion_tokens))
        prompt_tokens = int(usage.get("input_tokens", prompt_tokens))
        actual_cost = self._estimate_cost(prompt_tokens, completion_tokens)

        try:
            parsed = json.loads(raw_text)
        except Exception:
            decision = self._fallback("invalid_json")
            decision = GovernanceDecision(
                **{
                    **decision.__dict__,
                    "latency_ms": latency_ms,
                    "estimated_cost_usd": estimated_cost,
                    "actual_cost_usd": actual_cost,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
            self._record_usage(
                signal=signal,
                status="invalid_json",
                error="invalid_json",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost,
                actual_cost_usd=actual_cost,
                latency_ms=latency_ms,
            )
            self._record_failure()
            self._record_governance_log(signal, request_payload, decision, "invalid_json")
            return decision

        decision = GovernanceDecision(
            approve=bool(parsed.get("approve", True)),
            size_mult=_clamp(parsed.get("size_mult", 1.0), 0.10, 2.00),
            conf_adj=_clamp(parsed.get("conf_adj", 0.0), -0.30, 0.30),
            risk_mode=(
                str(parsed.get("risk_mode", "neutral")).lower()
                if str(parsed.get("risk_mode", "neutral")).lower() in self._RISK_MODES
                else "neutral"
            ),
            code=str(parsed.get("code", "LLM_OK"))[:64],
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost,
            actual_cost_usd=actual_cost,
            raw_response=parsed,
        )

        with self._lock:
            self._daily_spend_usd += actual_cost
            ttl_deadline = time.time() + max(1, settings.governance.cache_ttl_seconds)
            self._cache[cache_key] = (ttl_deadline, decision)
            self._failure_count = 0

        self._record_usage(
            signal=signal,
            status="success",
            error=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost,
            actual_cost_usd=actual_cost,
            latency_ms=latency_ms,
        )
        self._record_governance_log(signal, request_payload, decision, None)
        return decision

    def record_audit(
        self,
        signal: "TradeSignal",
        ppo_size_mult: float,
        governance_decision: GovernanceDecision,
        final_ai_score: float,
    ) -> None:
        session = get_session()
        try:
            session.add(
                AIDecisionAudit(
                    timestamp=datetime.utcnow(),
                    pair=signal.pair,
                    base_ai_score=float(signal.base_ai_score),
                    ppo_size_mult=float(ppo_size_mult),
                    governance_size_mult=float(governance_decision.size_mult),
                    governance_conf_adj=float(governance_decision.conf_adj),
                    final_ai_score=float(final_ai_score),
                    approved=bool(governance_decision.approve),
                    governance_code=str(governance_decision.code),
                    volatility_regime=str(signal.volatility_regime or "unknown"),
                    rsi_bucket=self._to_rsi_bucket(signal.features_snapshot.get("rsi_14", 50.0)),
                    metadata_json={
                        "risk_mode": governance_decision.risk_mode,
                        "from_cache": governance_decision.from_cache,
                        "fallback_reason": governance_decision.fallback_reason,
                        "latency_ms": governance_decision.latency_ms,
                    },
                )
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning(f"Failed to persist ai_decision_audit: {exc}")
        finally:
            session.close()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _request_llm(self, prompt: str) -> Dict[str, Any]:
        if not settings.governance.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is empty")

        payload = {
            "model": settings.governance.model,
            "max_tokens": settings.governance.max_output_tokens,
            "temperature": max(0.0, min(1.0, settings.governance.temperature)),
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Return JSON only. No markdown. "
                        "Required keys: approve,size_mult,conf_adj,risk_mode,code.\n"
                        f"{prompt}"
                    ),
                }
            ],
        }
        headers = {
            "x-api-key": settings.governance.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        timeout = max(0.1, float(settings.governance.timeout_seconds))

        start = time.perf_counter()
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                settings.governance.api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        raw_text = ""
        content = data.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    raw_text += str(block.get("text", ""))
        raw_text = raw_text.strip()
        return {
            "raw_text": raw_text,
            "usage": data.get("usage") or {},
            "latency_ms": (time.perf_counter() - start) * 1000.0,
        }

    def _build_request_payload(
        self,
        signal: "TradeSignal",
        ppo_size_mult: float,
        volatility_regime: str,
        rsi_bucket: str,
    ) -> Dict[str, Any]:
        return {
            "symbol": signal.pair,
            "side": signal.side,
            "base_ai_score": round(float(signal.base_ai_score), 6),
            "confidence": round(float(signal.confidence), 6),
            "ppo_size_mult": round(float(ppo_size_mult), 6),
            "volatility_regime": volatility_regime,
            "rsi_bucket": rsi_bucket,
            "market_regime": str(signal.market_regime or "unknown"),
            "risk_reward": round(float(signal.risk_reward), 6),
            "spread_pct": round(float(signal.spread_pct), 6),
            "atr_pct": (
                round(float(signal.atr / signal.entry_price), 6)
                if float(signal.entry_price) > 0
                else 0.0
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _to_rsi_bucket(self, rsi_value: Any) -> str:
        value = float(rsi_value or 50.0)
        if value < 30:
            return "lt30"
        if value < 45:
            return "30_45"
        if value < 55:
            return "45_55"
        if value < 70:
            return "55_70"
        return "gte70"

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        in_cost = (max(0, int(prompt_tokens)) / 1000.0) * settings.governance.input_cost_per_1k_tokens
        out_cost = (max(0, int(completion_tokens)) / 1000.0) * settings.governance.output_cost_per_1k_tokens
        return float(in_cost + out_cost)

    def _is_cost_blocked(self, estimated_cost: float) -> bool:
        with self._lock:
            if estimated_cost > settings.governance.per_trade_cost_cap:
                return True
            projected = self._daily_spend_usd + estimated_cost
            return projected > settings.governance.daily_cost_cap

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= settings.governance.max_failures_before_disable:
                self._auto_disabled = True
                logger.error("[GOV] Auto-disabled after repeated LLM failures")

    def _reset_daily_counters_if_needed(self) -> None:
        today = datetime.now().date()
        if today == self._active_date:
            return
        with self._lock:
            self._active_date = today
            self._failure_count = 0
            self._auto_disabled = False
            self._cache.clear()
            self._daily_spend_usd = self._query_today_spend()

    def _query_today_spend(self) -> float:
        session = get_session()
        try:
            now = datetime.utcnow()
            day_start = datetime(now.year, now.month, now.day)
            total = (
                session.query(func.sum(LLMUsage.actual_cost_usd))
                .filter(LLMUsage.created_at >= day_start, LLMUsage.status == "success")
                .scalar()
            )
            return float(total or 0.0)
        finally:
            session.close()

    def _get_cached(self, cache_key: Tuple[str, str, str]) -> GovernanceDecision | None:
        now = time.time()
        with self._lock:
            cached = self._cache.get(cache_key)
            if not cached:
                return None
            expires_at, value = cached
            if expires_at <= now:
                self._cache.pop(cache_key, None)
                return None
            return value

    def _fallback(self, reason: str) -> GovernanceDecision:
        return GovernanceDecision(
            approve=bool(FALLBACK_RESPONSE["approve"]),
            size_mult=float(FALLBACK_RESPONSE["size_mult"]),
            conf_adj=float(FALLBACK_RESPONSE["conf_adj"]),
            risk_mode=str(FALLBACK_RESPONSE["risk_mode"]),
            code=str(FALLBACK_RESPONSE["code"]),
            fallback_reason=reason,
            raw_response=dict(FALLBACK_RESPONSE),
        )

    def _record_governance_log(
        self,
        signal: "TradeSignal",
        request_payload: Dict[str, Any],
        decision: GovernanceDecision,
        error: str | None,
    ) -> None:
        session = get_session()
        try:
            session.add(
                GovernanceLog(
                    timestamp=datetime.utcnow(),
                    pair=signal.pair,
                    volatility_regime=str(signal.volatility_regime or "unknown"),
                    rsi_bucket=self._to_rsi_bucket(signal.features_snapshot.get("rsi_14", 50.0)),
                    request_payload=request_payload,
                    response_payload=decision.raw_response or dict(FALLBACK_RESPONSE),
                    approved=decision.approve,
                    size_mult=decision.size_mult,
                    conf_adj=decision.conf_adj,
                    risk_mode=decision.risk_mode,
                    code=decision.code,
                    latency_ms=decision.latency_ms,
                    from_cache=decision.from_cache,
                    fallback_reason=decision.fallback_reason,
                    error=error,
                )
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning(f"Failed to persist governance_logs: {exc}")
        finally:
            session.close()

    def _record_usage(
        self,
        signal: "TradeSignal",
        status: str,
        error: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost_usd: float,
        actual_cost_usd: float,
        latency_ms: float,
    ) -> None:
        session = get_session()
        try:
            session.add(
                LLMUsage(
                    timestamp=datetime.utcnow(),
                    provider=settings.governance.provider,
                    model=settings.governance.model,
                    pair=signal.pair,
                    prompt_tokens=max(0, int(prompt_tokens)),
                    completion_tokens=max(0, int(completion_tokens)),
                    estimated_cost_usd=max(0.0, float(estimated_cost_usd)),
                    actual_cost_usd=max(0.0, float(actual_cost_usd)),
                    status=status,
                    error=error,
                    latency_ms=max(0.0, float(latency_ms)),
                    created_at=datetime.utcnow(),
                )
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning(f"Failed to persist llm_usage: {exc}")
        finally:
            session.close()
