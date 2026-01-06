"""Schema migration helpers for the SQLite store."""

from __future__ import annotations

import sqlite3

from ..logging_utils import get_logger
from .sqlite_store import init_schema

_LOG = get_logger("migrations")


def _ensure_schema_version(conn: sqlite3.Connection) -> int:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        )
        """
    )
    cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
    row = cursor.fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version(version) VALUES (0)")
        conn.commit()
        return 0
    return int(row[0])


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute("UPDATE schema_version SET version = ?", (version,))
    conn.commit()


def apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply incremental migrations to reach the latest schema version."""

    current_version = _ensure_schema_version(conn)
    migrations = [init_schema]
    target_version = len(migrations)

    if current_version > target_version:
        raise RuntimeError(
            f"Database schema version {current_version} is newer than code {target_version}."
        )

    for idx in range(current_version, target_version):
        migration_number = idx + 1
        _LOG.info("Applying migration {}", migration_number)
        migrations[idx](conn)
        _set_schema_version(conn, migration_number)

    _LOG.info("SQLite schema at version {}", target_version)
