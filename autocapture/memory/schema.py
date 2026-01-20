"""SQLite schema management for the deterministic memory store."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 2


def ensure_schema(conn: sqlite3.Connection, *, require_fts: bool) -> bool:
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = cur.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        cur.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
    elif int(row[0]) < SCHEMA_VERSION:
        if int(row[0]) < 2:
            _ensure_hotness_tables(cur)
        cur.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            source_uri TEXT,
            title TEXT,
            content_type TEXT,
            payload_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            redaction_json TEXT,
            labels_json TEXT,
            excluded INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL,
            title TEXT,
            source_uri TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            labels_json TEXT,
            FOREIGN KEY(artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS document_text (
            doc_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS spans (
            span_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            section_path TEXT,
            text TEXT NOT NULL,
            span_sha256 TEXT NOT NULL,
            labels_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_items (
            item_id TEXT PRIMARY KEY,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            item_type TEXT NOT NULL,
            status TEXT NOT NULL,
            tags_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            value_sha256 TEXT NOT NULL,
            user_asserted INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_item_sources (
            item_id TEXT NOT NULL,
            span_id TEXT NOT NULL,
            PRIMARY KEY (item_id, span_id),
            FOREIGN KEY(item_id) REFERENCES memory_items(item_id) ON DELETE CASCADE,
            FOREIGN KEY(span_id) REFERENCES spans(span_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS context_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            created_at TEXT NOT NULL,
            config_sha256 TEXT NOT NULL,
            output_sha256 TEXT NOT NULL,
            retrieval_disabled INTEGER NOT NULL DEFAULT 0,
            span_count INTEGER NOT NULL DEFAULT 0,
            item_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshot_spans (
            snapshot_id TEXT NOT NULL,
            span_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            PRIMARY KEY (snapshot_id, span_id),
            FOREIGN KEY(snapshot_id) REFERENCES context_snapshots(snapshot_id) ON DELETE CASCADE,
            FOREIGN KEY(span_id) REFERENCES spans(span_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_spans_doc_id ON spans(doc_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_status ON memory_items(status)")
    _ensure_hotness_tables(cur)

    fts_available = True
    try:
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS spans_fts
            USING fts5(
                span_id UNINDEXED,
                doc_id UNINDEXED,
                title,
                section_path,
                text
            )
            """
        )
    except sqlite3.OperationalError:
        fts_available = False
        if require_fts:
            raise
    conn.commit()
    return fts_available


def _ensure_hotness_tables(cur: sqlite3.Cursor) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_hotness_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            item_id TEXT NOT NULL,
            ts_utc TEXT NOT NULL,
            signal TEXT NOT NULL,
            weight REAL NOT NULL,
            source TEXT NOT NULL,
            FOREIGN KEY(item_id) REFERENCES memory_items(item_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_hotness_state (
            scope TEXT NOT NULL,
            item_id TEXT NOT NULL,
            last_ts_utc TEXT NOT NULL,
            h_fast REAL NOT NULL,
            h_mid REAL NOT NULL,
            h_warm REAL NOT NULL,
            h_cool REAL NOT NULL,
            PRIMARY KEY (scope, item_id),
            FOREIGN KEY(item_id) REFERENCES memory_items(item_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_hotness_pins (
            scope TEXT NOT NULL,
            item_id TEXT NOT NULL,
            pin_level TEXT NOT NULL,
            pin_rank INTEGER NOT NULL,
            set_ts_utc TEXT NOT NULL,
            PRIMARY KEY (scope, item_id),
            FOREIGN KEY(item_id) REFERENCES memory_items(item_id) ON DELETE CASCADE
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hotness_events_scope_item_source_ts
        ON memory_hotness_events(scope, item_id, source, ts_utc, event_id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hotness_events_scope_ts
        ON memory_hotness_events(scope, ts_utc, event_id)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hotness_state_scope_last_ts
        ON memory_hotness_state(scope, last_ts_utc)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hotness_pins_scope_level_rank
        ON memory_hotness_pins(scope, pin_level, pin_rank, item_id)
        """
    )
