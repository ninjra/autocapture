"""Durable job queue stored in SQLite."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from ..logging_utils import get_logger
from .migrations import apply_migrations
from .sqlite_store import open_db

_LOG = get_logger("jobs")

_CONNECTION: sqlite3.Connection | None = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _get_connection() -> sqlite3.Connection:
    global _CONNECTION
    if _CONNECTION is None:
        data_dir = os.environ.get("AUTOCAPTURE_DATA_DIR", "./data")
        conn = open_db(data_dir)
        apply_migrations(conn)
        _CONNECTION = conn
    return _CONNECTION


def enqueue(kind: str, payload: dict[str, Any], priority: int = 0) -> int:
    """Insert a new job into the queue."""

    conn = _get_connection()
    payload_json = json.dumps(payload, ensure_ascii=False)
    now = _now_ms()
    cursor = conn.execute(
        """
        INSERT INTO jobs(kind, payload_json, priority, status, lease_until, attempts, last_error, created_at, updated_at)
        VALUES (?, ?, ?, 'pending', 0, 0, NULL, ?, ?)
        """,
        (kind, payload_json, priority, now, now),
    )
    conn.commit()
    job_id = int(cursor.lastrowid)
    _LOG.info(
        "Enqueued job {} ({}) priority={} payload_bytes={}",
        job_id,
        kind,
        priority,
        len(payload_json),
    )
    return job_id


def lease_one(kinds: list[str] | None, lease_ms: int):
    """Lease the next available job for processing."""

    conn = _get_connection()
    start = time.perf_counter()
    now = _now_ms()
    conn.execute("BEGIN IMMEDIATE")
    try:
        params: list[Any] = [now]
        kind_clause = ""
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            kind_clause = f"AND kind IN ({placeholders})"
            params.extend(kinds)
        params.extend([now])
        cursor = conn.execute(
            f"""
            SELECT id, kind, payload_json, priority, status, lease_until, attempts, created_at, updated_at
            FROM jobs
            WHERE (status = 'pending' OR status = 'retry')
              AND lease_until < ?
              {kind_clause}
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            """,
            params,
        )
        row = cursor.fetchone()
        if row is None:
            conn.commit()
            return None
        lease_until = now + lease_ms
        conn.execute(
            """
            UPDATE jobs
            SET status = 'running', lease_until = ?, attempts = attempts + 1, updated_at = ?
            WHERE id = ?
            """,
            (lease_until, now, row["id"]),
        )
        conn.commit()
        duration_ms = (time.perf_counter() - start) * 1000
        _LOG.info(
            "Leased job {} ({}) in {:.1f}ms",
            row["id"],
            row["kind"],
            duration_ms,
        )
        return dict(row)
    except Exception:
        conn.rollback()
        raise


def _get_job_meta(
    conn: sqlite3.Connection, job_id: int
) -> tuple[str | None, int | None]:
    cursor = conn.execute(
        "SELECT kind, updated_at FROM jobs WHERE id = ?",
        (job_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None, None
    return row["kind"], row["updated_at"]


def mark_done(job_id: int) -> None:
    """Mark a job as successfully completed."""

    conn = _get_connection()
    kind, updated_at = _get_job_meta(conn, job_id)
    now = _now_ms()
    conn.execute(
        """
        UPDATE jobs
        SET status = 'done', lease_until = 0, updated_at = ?
        WHERE id = ?
        """,
        (now, job_id),
    )
    conn.commit()
    duration_ms = now - updated_at if updated_at else None
    _LOG.info("Job {} ({}) done duration_ms={}", job_id, kind, duration_ms)


def mark_retry(job_id: int, error: str, backoff_ms: int) -> None:
    """Mark a job as retryable with backoff."""

    conn = _get_connection()
    kind, updated_at = _get_job_meta(conn, job_id)
    now = _now_ms()
    conn.execute(
        """
        UPDATE jobs
        SET status = 'retry', lease_until = ?, last_error = ?, updated_at = ?
        WHERE id = ?
        """,
        (now + backoff_ms, error, now, job_id),
    )
    conn.commit()
    duration_ms = now - updated_at if updated_at else None
    _LOG.warning(
        "Job {} ({}) retrying duration_ms={} backoff_ms={} error={}",
        job_id,
        kind,
        duration_ms,
        backoff_ms,
        error,
    )


def mark_failed(job_id: int, error: str) -> None:
    """Mark a job as permanently failed."""

    conn = _get_connection()
    kind, updated_at = _get_job_meta(conn, job_id)
    now = _now_ms()
    conn.execute(
        """
        UPDATE jobs
        SET status = 'failed', lease_until = 0, last_error = ?, updated_at = ?
        WHERE id = ?
        """,
        (error, now, job_id),
    )
    conn.commit()
    duration_ms = now - updated_at if updated_at else None
    _LOG.error(
        "Job {} ({}) failed duration_ms={} error={}", job_id, kind, duration_ms, error
    )
