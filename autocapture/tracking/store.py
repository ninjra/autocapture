"""SQLite persistence for host vector events."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
import os
from typing import Sequence

from ..config import TrackingConfig
from ..logging_utils import get_logger
from .types import HostEventRow, RawInputRow


def resolve_tracking_db_path(config: TrackingConfig, data_dir: Path | None) -> Path:
    db_path = config.db_path
    if not db_path.is_absolute():
        if data_dir is None:
            return Path(db_path)
        return (Path(data_dir) / db_path).resolve()
    return Path(db_path)


class SqliteHostEventStore:
    """Lightweight SQLite store for host vector events."""

    def __init__(self, db_path: Path, *, config: TrackingConfig | None = None) -> None:
        self._db_path = db_path
        self._log = get_logger("tracking.store")
        self._config = config
        connect_args = {"check_same_thread": False}
        if config and config.encryption_enabled:
            self._conn = self._connect_sqlcipher(connect_args)
        else:
            self._conn = sqlite3.connect(self._db_path, **connect_args)
        self._conn.row_factory = sqlite3.Row
        self._apply_pragmas()

    def _apply_pragmas(self) -> None:
        cur = self._conn.cursor()
        if self._config and self._config.encryption_enabled:
            key = self._load_key()
            cur.execute("PRAGMA key = ?", (key,))
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    def close(self) -> None:
        self._conn.close()

    def init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS host_events (
              id TEXT PRIMARY KEY,
              ts_start_ms INTEGER NOT NULL,
              ts_end_ms INTEGER NOT NULL,
              kind TEXT NOT NULL,
              session_id TEXT,
              app_name TEXT,
              window_title TEXT,
              payload_json TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS host_input_events (
              id TEXT PRIMARY KEY,
              ts_ms INTEGER NOT NULL,
              monotonic_ms INTEGER NOT NULL,
              device TEXT NOT NULL,
              kind TEXT NOT NULL,
              session_id TEXT,
              app_name TEXT,
              window_title TEXT,
              payload_json TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_host_events_ts_start ON host_events(ts_start_ms);"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_host_events_kind ON host_events(kind);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_host_events_app ON host_events(app_name);")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_host_input_events_ts ON host_input_events(ts_ms);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_host_input_events_kind ON host_input_events(kind);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_host_input_events_app ON host_input_events(app_name);"
        )
        self._conn.commit()
        cursor.close()
        self._log.info("Host events schema initialized")

    def insert_many(self, rows: Sequence[HostEventRow]) -> None:
        if not rows:
            return
        start = time.perf_counter()
        payload = [
            (
                row.id,
                row.ts_start_ms,
                row.ts_end_ms,
                row.kind,
                row.session_id,
                row.app_name,
                row.window_title,
                row.payload_json,
            )
            for row in rows
        ]
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO host_events (
                  id, ts_start_ms, ts_end_ms, kind, session_id, app_name, window_title, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._log.info(
            "Inserted {} host events in {:.2f}ms",
            len(rows),
            elapsed_ms,
        )

    def insert_raw_events(self, rows: Sequence[RawInputRow]) -> None:
        if not rows:
            return
        start = time.perf_counter()
        payload = [
            (
                row.id,
                row.ts_ms,
                row.monotonic_ms,
                row.device,
                row.kind,
                row.session_id,
                row.app_name,
                row.window_title,
                row.payload_json,
            )
            for row in rows
        ]
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO host_input_events (
                  id, ts_ms, monotonic_ms, device, kind,
                  session_id, app_name, window_title, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._log.info(
            "Inserted {} raw input events in {:.2f}ms",
            len(rows),
            elapsed_ms,
        )

    def prune_older_than(self, cutoff_ms: int) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM host_events WHERE ts_end_ms < ?",
                (cutoff_ms,),
            )
        deleted = cursor.rowcount or 0
        self._log.info("Pruned {} host events older than {}", deleted, cutoff_ms)
        return deleted

    def prune_raw_older_than(self, cutoff_ms: int) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM host_input_events WHERE ts_ms < ?",
                (cutoff_ms,),
            )
        deleted = cursor.rowcount or 0
        self._log.info("Pruned {} raw input events older than {}", deleted, cutoff_ms)
        return deleted

    def query_recent(self, limit: int = 50) -> list[sqlite3.Row]:
        cursor = self._conn.execute(
            "SELECT * FROM host_events ORDER BY ts_start_ms DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def query_raw_events(
        self,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 200,
        offset: int = 0,
        device: str | None = None,
        kind: str | None = None,
        app_name: str | None = None,
        window_title: str | None = None,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[object] = []
        if start_ms is not None:
            clauses.append("ts_ms >= ?")
            params.append(start_ms)
        if end_ms is not None:
            clauses.append("ts_ms <= ?")
            params.append(end_ms)
        if device:
            clauses.append("device = ?")
            params.append(device)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if app_name:
            clauses.append("app_name = ?")
            params.append(app_name)
        if window_title:
            clauses.append("window_title = ?")
            params.append(window_title)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = "SELECT * FROM host_input_events " f"{where} ORDER BY ts_ms ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def _connect_sqlcipher(self, connect_args: dict) -> sqlite3.Connection:
        try:
            import pysqlcipher3.dbapi2 as sqlcipher  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "SQLCipher support requires pysqlcipher3. Install via: poetry install --extras sqlcipher "
                "(Windows uses rotki-pysqlcipher3 wheels)."
            ) from exc
        return sqlcipher.connect(self._db_path, **connect_args)

    def _load_key(self) -> bytes:
        if not self._config:
            raise RuntimeError("Tracking config required for encryption")
        provider = self._config.encryption_key_provider
        if provider == "env":
            value = os.getenv(self._config.encryption_env_var)
            if not value:
                raise RuntimeError("Tracking encryption env var missing")
            return bytes.fromhex(value)
        if provider not in {"file", "dpapi_file"}:
            raise ValueError(f"Unsupported tracking encryption provider: {provider}")
        path = self._config.encryption_key_path
        if not path.is_absolute():
            path = self._db_path.parent / path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            key = os.urandom(32)
            if provider == "dpapi_file":
                try:
                    import win32crypt  # pragma: no cover - Windows specific

                    protected = win32crypt.CryptProtectData(key, None, None, None, None, 0)
                    path.write_bytes(protected)
                except Exception:
                    path.write_bytes(key)
            else:
                path.write_bytes(key)
            _ensure_private_permissions(path)
            return key
        data = path.read_bytes()
        if provider == "dpapi_file":
            try:
                import win32crypt  # pragma: no cover - Windows specific

                data = win32crypt.CryptUnprotectData(data, None, None, None, 0)[1]
            except Exception:
                pass
        _ensure_private_permissions(path)
        return data


def _ensure_private_permissions(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o600)
    except Exception:
        return


def safe_payload(payload: dict) -> str:
    """Serialize payloads to JSON without leaking secrets in logs."""

    return json.dumps(payload, separators=(",", ":"), sort_keys=True)
