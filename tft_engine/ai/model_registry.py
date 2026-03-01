"""
Model registry service backed by SQLite table model_registry.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from tft_engine.database.models import ModelRegistry

logger = logging.getLogger(__name__)


class ModelRegistryService:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def register_model(
        self,
        model_type: str,
        version: str,
        path: str,
        metrics: Optional[dict[str, Any]] = None,
    ) -> ModelRegistry:
        session: Session = self._session_factory()
        try:
            record = ModelRegistry(
                model_type=model_type,
                version=version,
                path=path,
                created_at=datetime.utcnow(),
                metrics_json=json.dumps(metrics or {}),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info(f"Registered model {model_type}:{version} at {path}")
            return record
        finally:
            session.close()

    def latest(self, model_type: str) -> Optional[ModelRegistry]:
        session: Session = self._session_factory()
        try:
            return (
                session.query(ModelRegistry)
                .filter(ModelRegistry.model_type == model_type)
                .order_by(desc(ModelRegistry.created_at), desc(ModelRegistry.id))
                .first()
            )
        finally:
            session.close()
