"""
Structured JSON logging with loguru.
All components use this for consistent, parseable log output.
"""
from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from config.settings import settings


def _json_sink(message: Any) -> None:
    """Write structured JSON to log files."""
    record = message.record
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    # Merge any extra fields
    if record.get("extra"):
        for k, v in record["extra"].items():
            if k not in ("_serialized",):
                log_entry[k] = v
    print(json.dumps(log_entry), file=sys.stderr, flush=True)


def setup_logging() -> None:
    """Configure logging for the entire application."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    log_dir = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console - structured JSON to stdout
    logger.add(
        sys.stdout,
        level=settings.log_level,
        serialize=True,
    )

    # JSON file - structured, parseable
    logger.add(
        str(log_dir / "engine_{time:YYYY-MM-DD}.json"),
        level="DEBUG",
        format="{message}",
        serialize=True,
        rotation="1 day",
        retention="30 days",
        compression="gz",
    )

    # Production rotating logs (20MB/file, 10 backups)
    rotating_handler = RotatingFileHandler(
        str(log_dir / "engine.log"),
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    rotating_handler.setLevel(logging.INFO)
    logger.add(
        rotating_handler,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        enqueue=True,
    )

    # Trade-specific log
    logger.add(
        str(log_dir / "trades_{time:YYYY-MM-DD}.json"),
        level="INFO",
        filter=lambda record: record["extra"].get("log_type") == "trade",
        serialize=True,
        rotation="1 day",
        retention="90 days",
    )

    # Error log
    logger.add(
        str(log_dir / "errors_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        rotation="1 day",
        retention="60 days",
    )


def log_trade(data: Dict[str, Any]) -> None:
    """Log a trade event with structured data."""
    logger.bind(log_type="trade", **data).info("Trade event: {event}", event=data.get("event", "unknown"))


def log_signal(data: Dict[str, Any]) -> None:
    """Log a signal/forecast event."""
    logger.bind(log_type="signal", **data).info("Signal: {pair} conf={confidence:.3f}", 
                                                  pair=data.get("pair", "?"),
                                                  confidence=data.get("confidence", 0))


def log_api_error(endpoint: str, error: str, **kwargs: Any) -> None:
    """Log an API error."""
    logger.bind(log_type="api_error", endpoint=endpoint, error=error, **kwargs).error(
        "API error on {endpoint}: {error}", endpoint=endpoint, error=error
    )
