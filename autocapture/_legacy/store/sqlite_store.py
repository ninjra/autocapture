"""SQLite data store for capture metadata."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..logging_utils import get_logger

_LOG = get_logger("sqlite")


def open_db(data_dir: Path | str) -> sqlite3.Connection:
    """Open a SQLite connection configured for concurrent reads/writes."""

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    db_path = data_path / "autocapture.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _LOG.info("Opened SQLite DB at {}", db_path)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create base tables for segments, observations, jobs, and search."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS segments (
            id TEXT PRIMARY KEY,
            started_at INTEGER NOT NULL,
            ended_at INTEGER,
            layout_json TEXT NOT NULL,
            video_path TEXT,
            state TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS observations (
            id TEXT PRIMARY KEY,
            segment_id TEXT NOT NULL,
            ts INTEGER NOT NULL,
            monitor_id TEXT NOT NULL,
            cursor_x INTEGER NOT NULL,
            cursor_y INTEGER NOT NULL,
            roi_path TEXT NOT NULL,
            thumb_path TEXT,
            fg_process TEXT,
            fg_title TEXT,
            ocr_text TEXT,
            vision_summary TEXT,
            FOREIGN KEY(segment_id) REFERENCES segments(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            lease_until INTEGER DEFAULT 0,
            attempts INTEGER DEFAULT 0,
            last_error TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS segment_fts
        USING fts5(segment_id, started_at, ended_at, content)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_observations_segment_ts
        ON observations(segment_id, ts)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_status_lease
        ON jobs(status, lease_until, priority, created_at)
        """
    )
    conn.commit()
    _LOG.info("SQLite schema initialised")


def upsert_segment_fts(
    conn: sqlite3.Connection,
    segment_id: str,
    started_at: int,
    ended_at: int | None,
    content: str,
) -> None:
    """Replace the FTS row for a segment with aggregated text."""

    conn.execute("DELETE FROM segment_fts WHERE segment_id = ?", (segment_id,))
    conn.execute(
        """
        INSERT INTO segment_fts(segment_id, started_at, ended_at, content)
        VALUES (?, ?, ?, ?)
        """,
        (segment_id, started_at, ended_at, content),
    )
    conn.commit()
    _LOG.info("segment_fts updated for segment {} (chars={})", segment_id, len(content))
