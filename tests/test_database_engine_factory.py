from __future__ import annotations

import importlib

import pytest
from sqlalchemy import text


_DB_ENV_KEYS = [
    "DATABASE_MODE",
    "SQLITE_PATH",
    "SQLITE_WAL_MODE",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_POOL_SIZE",
    "POSTGRES_MAX_OVERFLOW",
]


def _reload_database_modules(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]):
    for key in _DB_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))

    import config.settings as settings_module
    import data.database as database_module

    settings_module = importlib.reload(settings_module)
    database_module = importlib.reload(database_module)
    return settings_module, database_module


def test_sqlite_engine_creation_enables_wal_and_pragmas(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    db_file = tmp_path / "engine.sqlite3"
    _, database_module = _reload_database_modules(
        monkeypatch,
        {
            "DATABASE_MODE": "SQLITE",
            "SQLITE_PATH": str(db_file),
            "SQLITE_WAL_MODE": "true",
        },
    )

    engine = database_module.get_engine()
    assert engine.url.get_backend_name() == "sqlite"

    with engine.connect() as conn:
        journal_mode = conn.execute(text("PRAGMA journal_mode;")).scalar_one()
        synchronous_mode = conn.execute(text("PRAGMA synchronous;")).scalar_one()
        temp_store = conn.execute(text("PRAGMA temp_store;")).scalar_one()
        cache_size = conn.execute(text("PRAGMA cache_size;")).scalar_one()

    assert str(journal_mode).lower() == "wal"
    assert int(synchronous_mode) == 1  # NORMAL
    assert int(temp_store) == 2  # MEMORY
    assert int(cache_size) == 100000

    session = database_module.get_session()
    try:
        assert session.autoflush is False
    finally:
        session.close()
        database_module.dispose_engine()


def test_postgres_mode_without_psycopg2_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    _, database_module = _reload_database_modules(
        monkeypatch,
        {
            "DATABASE_MODE": "POSTGRES",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "tft_trading",
            "POSTGRES_USER": "trader",
            "POSTGRES_PASSWORD": "secret",
        },
    )

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "psycopg2":
            raise ModuleNotFoundError("No module named 'psycopg2'")
        return real_import_module(name, package)

    monkeypatch.setattr(database_module.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="requires psycopg2"):
        database_module.get_engine()
    database_module.dispose_engine()


def test_postgres_connection_failure_does_not_fallback_to_sqlite(
    monkeypatch: pytest.MonkeyPatch,
):
    _, database_module = _reload_database_modules(
        monkeypatch,
        {
            "DATABASE_MODE": "POSTGRES",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "tft_trading",
            "POSTGRES_USER": "trader",
            "POSTGRES_PASSWORD": "secret",
        },
    )

    called = {"sqlite_factory_called": False}

    def fail_if_called():
        called["sqlite_factory_called"] = True
        raise AssertionError("SQLite factory should never be called in POSTGRES mode.")

    def fail_postgres_engine():
        raise Exception("connection failed")

    monkeypatch.setattr(database_module, "_ensure_psycopg2_installed", lambda: None)
    monkeypatch.setattr(database_module, "_create_sqlite_engine", fail_if_called)
    monkeypatch.setattr(database_module, "_create_postgres_engine", fail_postgres_engine)

    with pytest.raises(RuntimeError, match="PostgreSQL unavailable. Fix configuration."):
        database_module.get_engine()

    assert called["sqlite_factory_called"] is False
    database_module.dispose_engine()
