"""
Database connection utilities for SQLite (default) and PostgreSQL (optional).
"""
from __future__ import annotations

import atexit
import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

BASE_DIR = Path(__file__).resolve().parents[2]


def _database_mode() -> str:
    mode = os.getenv("DATABASE_MODE", "SQLITE").strip().upper()
    if mode not in {"SQLITE", "POSTGRES"}:
        raise ValueError(
            f"Unknown DATABASE_MODE '{mode}'. Expected SQLITE or POSTGRES."
        )
    return mode


def _sqlite_path() -> Path:
    value = os.getenv("SQLITE_PATH", "./data/tft_engine.db").strip()
    path = Path(value)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sqlite_wal_mode() -> bool:
    return os.getenv("SQLITE_WAL_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}


def _postgres_url() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost").strip() or "localhost"
    port = os.getenv("POSTGRES_PORT", "5432").strip() or "5432"
    db = os.getenv("POSTGRES_DB", "tft_trading").strip() or "tft_trading"
    user = os.getenv("POSTGRES_USER", "trader").strip() or "trader"
    password = os.getenv("POSTGRES_PASSWORD", "").strip()
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def _configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    if _sqlite_wal_mode():
        cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA temp_store=MEMORY;")
    cursor.execute("PRAGMA cache_size=100000;")
    cursor.close()


def _create_engine():
    mode = _database_mode()
    if mode == "SQLITE":
        sqlite_url = f"sqlite:///{_sqlite_path().as_posix()}"
        engine = create_engine(
            sqlite_url,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False},
            pool_size=1,
            max_overflow=0,
        )
        event.listen(engine, "connect", _configure_sqlite_connection)
        return engine

    try:
        importlib.import_module("psycopg2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DATABASE_MODE=POSTGRES requires psycopg2. Install psycopg2-binary or psycopg2."
        ) from exc

    pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "10").strip() or "10")
    max_overflow = int(os.getenv("POSTGRES_MAX_OVERFLOW", "20").strip() or "20")
    return create_engine(
        _postgres_url(),
        echo=False,
        future=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
    )


engine = _create_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def dispose_engine() -> None:
    engine.dispose()


atexit.register(dispose_engine)


def get_session() -> Session:
    return SessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
