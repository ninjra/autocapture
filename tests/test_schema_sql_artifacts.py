from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs" / "schemas"


def _execute_sql(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(path.read_text(encoding="utf-8"))
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if "fts5" in message:
            pytest.fail(f"FTS5 required to execute {path.name}; sqlite3 build lacks fts5 support.")
        raise
    return conn


@pytest.mark.parametrize(
    "filename, expected_tables",
    [
        (
            "autocapture_memory_store.sql",
            {
                "schema_version",
                "artifacts",
                "documents",
                "document_text",
                "spans",
                "memory_items",
                "memory_item_sources",
                "context_snapshots",
                "snapshot_spans",
                "memory_hotness_events",
                "memory_hotness_state",
                "memory_hotness_pins",
                "spans_fts",
            },
        ),
        (
            "autocapture_tracking_store.sql",
            {
                "host_events",
                "host_input_events",
            },
        ),
        (
            "sqlite_rag_core.sql",
            {
                "event_fts",
                "span_fts",
                "chunks_fts",
                "thread_fts",
            },
        ),
    ],
)
def test_schema_sql_executes_and_objects_exist(filename: str, expected_tables: set[str]) -> None:
    path = SCHEMA_DIR / filename
    assert path.exists(), f"Missing schema SQL artifact: {path}"
    conn = _execute_sql(path)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    found = {row[0] for row in rows}
    missing = expected_tables - found
    assert not missing, f"Missing expected tables in {filename}: {sorted(missing)}"
