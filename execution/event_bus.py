"""
Lightweight in-process event bus for execution pipeline telemetry.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from threading import Lock
from typing import Any, Deque, Dict, List

from loguru import logger

_MAX_EVENTS = 5000
_EVENTS: Deque[Dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
_LOCK = Lock()


def publish_event(event_type: str, payload: Dict[str, Any] | None = None) -> None:
    event = {
        "event_type": str(event_type or "UNKNOWN"),
        "timestamp": datetime.utcnow().isoformat(),
        "payload": payload or {},
    }
    with _LOCK:
        _EVENTS.append(event)
    logger.bind(event=event["event_type"], **event["payload"]).info(event["event_type"])


def recent_events(limit: int = 100) -> List[Dict[str, Any]]:
    size = max(1, min(int(limit), _MAX_EVENTS))
    with _LOCK:
        return list(_EVENTS)[-size:]
