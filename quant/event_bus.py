from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from loguru import logger


@dataclass
class QuantEvent:
    event_type: str
    timestamp: datetime
    payload: Dict[str, Any]


class AsyncEventBus:
    """In-process async event bus used by the quant orchestrator."""

    def __init__(self, max_events: int = 5000) -> None:
        self._queue: asyncio.Queue[QuantEvent] = asyncio.Queue()
        self._events: List[QuantEvent] = []
        self._max_events = max(100, int(max_events))

    async def publish(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        event = QuantEvent(
            event_type=str(event_type or "UNKNOWN"),
            timestamp=datetime.utcnow(),
            payload=payload or {},
        )
        await self._queue.put(event)
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]
        logger.bind(event=event.event_type, **event.payload).info(event.event_type)

    async def next_event(self, timeout_sec: float = 0.0) -> QuantEvent | None:
        if timeout_sec <= 0:
            if self._queue.empty():
                return None
            return await self._queue.get()
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            return None

    def recent(self, limit: int = 200) -> List[Dict[str, Any]]:
        size = max(1, min(int(limit), self._max_events))
        sliced = self._events[-size:]
        return [
            {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "payload": dict(event.payload),
            }
            for event in sliced
        ]
