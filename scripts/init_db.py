"""Initialize the database tables."""
import sys
sys.path.insert(0, ".")

from data.database import init_db
from utils.logging import setup_logging
from loguru import logger

if __name__ == "__main__":
    setup_logging()
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully.")
