from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

from autocapture.tracking.store import SqliteHostEventStore, safe_payload
from autocapture.tracking.types import HostEventRow, RawInputRow


def _row(ts_start: int, ts_end: int, kind: str) -> HostEventRow:
    return HostEventRow(
        id=str(uuid4()),
        ts_start_ms=ts_start,
        ts_end_ms=ts_end,
        kind=kind,
        session_id=None,
        app_name="demo.exe",
        window_title="Demo",
        payload_json=safe_payload({"count": 1}),
    )


def _raw_row(ts_ms: int, kind: str) -> RawInputRow:
    return RawInputRow(
        id=str(uuid4()),
        ts_ms=ts_ms,
        monotonic_ms=ts_ms,
        device="mouse",
        kind=kind,
        session_id=None,
        app_name="demo.exe",
        window_title="Demo",
        payload_json=safe_payload({"dx": 1, "dy": 2}),
    )


def test_store_insert_and_query(tmp_path: Path) -> None:
    db_path = tmp_path / "events.sqlite"
    store = SqliteHostEventStore(db_path)
    store.init_schema()
    rows = [_row(1000, 1001, "input_bucket"), _row(1100, 1100, "foreground_change")]
    store.insert_many(rows)
    fetched = store.query_recent(limit=10)
    store.close()

    assert len(fetched) == 2
    assert fetched[0]["kind"] in {"input_bucket", "foreground_change"}


def test_store_prune(tmp_path: Path) -> None:
    db_path = tmp_path / "events.sqlite"
    store = SqliteHostEventStore(db_path)
    store.init_schema()
    store.insert_many(
        [
            _row(1000, 1000, "input_bucket"),
            _row(2000, 2000, "input_bucket"),
        ]
    )
    deleted = store.prune_older_than(1500)
    remaining = store.query_recent(limit=10)
    store.close()

    assert deleted == 1
    assert len(remaining) == 1
    assert remaining[0]["ts_end_ms"] == 2000


def test_raw_event_insert_and_query(tmp_path: Path) -> None:
    db_path = tmp_path / "events.sqlite"
    store = SqliteHostEventStore(db_path)
    store.init_schema()
    store.insert_raw_events([_raw_row(1000, "mouse_move"), _raw_row(1500, "mouse_button")])
    rows = store.query_raw_events(start_ms=900, end_ms=1600, limit=10)
    store.close()

    assert len(rows) == 2
    assert rows[0]["kind"] in {"mouse_move", "mouse_button"}


def test_schema_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "events.sqlite"
    store = SqliteHostEventStore(db_path)
    store.init_schema()
    conn = sqlite3.connect(db_path)
    indexes = {row[1] for row in conn.execute("PRAGMA index_list('host_events')")}
    raw_indexes = {row[1] for row in conn.execute("PRAGMA index_list('host_input_events')")}
    conn.close()
    store.close()

    assert "idx_host_events_ts_start" in indexes
    assert "idx_host_events_kind" in indexes
    assert "idx_host_events_app" in indexes
    assert "idx_host_input_events_ts" in raw_indexes
    assert "idx_host_input_events_kind" in raw_indexes
    assert "idx_host_input_events_app" in raw_indexes
