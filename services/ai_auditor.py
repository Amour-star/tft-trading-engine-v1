"""AI audit guard that uses Gemini to double-check each trade."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from loguru import logger

AUDITOR_READY = False
genai: Any | None = None

AI_AUDITOR_ENABLED = os.getenv("AI_AUDITOR_ENABLED", "false").strip().lower() in {
    "true",
    "1",
    "yes",
}
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

try:
    import google.generativeai as genai_module

    if AI_AUDITOR_ENABLED:
        if not GEMINI_API_KEY:
            logger.warning("[AI_AUDIT] Gemini auditor enabled but GEMINI_API_KEY is missing")
        else:
            genai_module.configure(api_key=GEMINI_API_KEY)
            genai = genai_module
            AUDITOR_READY = True
except ImportError:  # pragma: no cover - dependency optional
    if AI_AUDITOR_ENABLED:
        logger.warning(
            "[AI_AUDIT] google.generativeai SDK unavailable, disabling auditor"
        )
except Exception as exc:  # pragma: no cover - catch runtime issues without bubbling
    logger.warning("[AI_AUDIT] Failed to initialize auditor ({}), disabling", exc)


def audit_trade(trade_data: Dict[str, Any]) -> Dict[str, Any]:
    """Ask Gemini for a sanity check before allowing high-risk trades."""
    if not AUDITOR_READY or not genai:
        return {"valid": True, "reason": "Auditor disabled"}

    payload = json.dumps(trade_data, sort_keys=True)
    prompt = f"""
    Analyze this crypto trade for logical consistency.
    Detect:
    - Zero or abnormal prices
    - Unrealistic quantity
    - Over-leverage
    - Scale mismatch
    - Risk too high
    Return JSON:
    {{ "valid": true/false, "reason": "..." }}
    Trade:
    {payload}
    """
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        text = _extract_text(response)
        audit = _parse_json(text)
        valid = bool(audit.get("valid", True))
        reason = str(audit.get("reason", "No reason provided."))
        logger.info(
            "[AI_AUDIT] {symbol} trade audit result: valid={valid} reason={reason}",
            symbol=trade_data.get("symbol"),
            valid=valid,
            reason=reason,
        )
        return {"valid": valid, "reason": reason}
    except Exception as exc:
        logger.warning("[AI_AUDIT] Audit failure ({}), failing open", exc)
        return {"valid": True, "reason": "Audit failed"}


def _extract_text(response: Any) -> str:
    """Robustly extract text from GenerativeModel output."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "text") and response.text:
        return str(response.text)
    if hasattr(response, "response"):
        resp = getattr(response, "response")
        if hasattr(resp, "text") and resp.text:
            return str(resp.text)
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, (list, tuple)) and candidates:
        parts = getattr(candidates[0], "content", None) or getattr(
            candidates[0], "parts", None
        )
        if isinstance(parts, list):
            for part in parts:
                text = _extract_text(part)
                if text:
                    return text
    return str(response)


def _parse_json(message: str) -> Dict[str, Any]:
    """Attempt to parse JSON from the first brace-enclosed blob."""
    trimmed = message.strip()
    if not trimmed:
        raise ValueError("Empty audit response")
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start >= 0 and end > start:
            snippet = trimmed[start : end + 1]
            return json.loads(snippet)
        raise
