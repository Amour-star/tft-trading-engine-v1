"""
Lightweight migration helpers for SQLite deployments.

For production upgrades, Alembic should be used. This file provides a safe
bootstrap path for environments where schema initialization is needed quickly.
"""
from __future__ import annotations

import logging
import os

from tft_engine.database.connection import engine
from tft_engine.database.models import Base

logger = logging.getLogger(__name__)


def _database_mode() -> str:
    mode = os.getenv("DATABASE_MODE", "SQLITE").strip().upper()
    if mode not in {"SQLITE", "POSTGRES"}:
        raise ValueError(
            f"Unknown DATABASE_MODE '{mode}'. Expected SQLITE or POSTGRES."
        )
    return mode


def initialize_database() -> None:
    mode = _database_mode()
    if mode == "SQLITE":
        logger.info("Initializing SQLite schema...")
        Base.metadata.create_all(bind=engine)
        logger.info("SQLite schema ready.")
        return
    logger.info(
        "DATABASE_MODE=POSTGRES selected. Automatic migrations are disabled; "
        "run PostgreSQL migrations explicitly."
    )


def reset_database() -> None:
    if _database_mode() != "SQLITE":
        raise RuntimeError("reset_database is supported only when DATABASE_MODE=SQLITE.")
    logger.warning("Resetting SQLite schema (drop + create).")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
