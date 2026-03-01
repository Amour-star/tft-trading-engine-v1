"""Initialize the database tables."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import init_db
from utils.logging import setup_logging
from loguru import logger

if __name__ == "__main__":
    setup_logging()
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully.")
