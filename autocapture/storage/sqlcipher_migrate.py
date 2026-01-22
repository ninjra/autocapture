"""SQLCipher migration helper."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from ..config import DatabaseConfig
from ..logging_utils import get_logger
from ..security.sqlcipher import load_sqlcipher_key


def encrypt_sqlite_database(config: DatabaseConfig) -> Path:
    if not config.encryption_enabled:
        raise RuntimeError("database.encryption_enabled must be true to run encryption migration")
    if not config.url.startswith("sqlite"):
        raise RuntimeError("SQLCipher migration only supports sqlite databases")
    source_path = Path(config.url.replace("sqlite:///", ""))
    if not source_path.exists():
        raise FileNotFoundError(f"Source database not found: {source_path}")
    if source_path.name == ":memory:":
        raise RuntimeError("Cannot encrypt in-memory sqlite database")
    log = get_logger("db.sqlcipher")
    key = load_sqlcipher_key(config, source_path.parent)
    backup_path = source_path.with_suffix(source_path.suffix + ".bak")
    temp_path = source_path.with_suffix(source_path.suffix + ".enc.tmp")
    if backup_path.exists():
        log.info("Backup already exists at {}", backup_path)
    else:
        shutil.copy2(source_path, backup_path)
        log.info("Backup created at {}", backup_path)
    if temp_path.exists():
        temp_path.unlink()
    dump_sql = _dump_sql(source_path)
    _write_encrypted(temp_path, dump_sql, key)
    _verify_encrypted(temp_path, key)
    temp_path.replace(source_path)
    log.info("Encrypted database written to {}", source_path)
    return source_path


def _dump_sql(path: Path) -> str:
    with sqlite3.connect(path) as conn:
        return "\n".join(conn.iterdump())


def _write_encrypted(path: Path, dump_sql: str, key: bytes) -> None:
    try:
        import pysqlcipher3.dbapi2 as sqlcipher  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SQLCipher support requires pysqlcipher3. Install via: poetry install --extras sqlcipher "
            "(Windows uses rotki-pysqlcipher3 wheels)."
        ) from exc
    with sqlcipher.connect(path) as conn:  # type: ignore
        _apply_sqlcipher_key(conn, key)
        conn.executescript(dump_sql)
        conn.commit()


def _verify_encrypted(path: Path, key: bytes) -> None:
    try:
        import pysqlcipher3.dbapi2 as sqlcipher  # type: ignore
    except Exception:  # pragma: no cover
        return
    with sqlcipher.connect(path) as conn:  # type: ignore
        _apply_sqlcipher_key(conn, key)
        conn.execute("SELECT name FROM sqlite_master LIMIT 1")


def _apply_sqlcipher_key(conn, key: bytes) -> None:
    try:
        conn.execute("PRAGMA key = ?", (key,))
        return
    except Exception as exc:
        message = str(exc).lower()
        if 'near "?"' not in message and "near '?'" not in message:
            raise
    hex_key = key.hex()
    conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")
