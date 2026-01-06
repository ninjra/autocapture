from __future__ import annotations

import datetime as dt
from pathlib import Path

from autocapture.config import DatabaseConfig
from autocapture.indexing.lexical_index import LexicalIndex
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord


def test_lexical_bulk_upsert(tmp_path: Path) -> None:
    db_path = tmp_path / "events.sqlite"
    db = DatabaseManager(DatabaseConfig(url=f"sqlite:///{db_path}"))
    event = EventRecord(
        event_id="event-1",
        ts_start=dt.datetime.now(dt.timezone.utc),
        ts_end=None,
        app_name="app",
        window_title="window",
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash",
        ocr_text="hello world",
        embedding_vector=None,
        embedding_status="pending",
        embedding_model=None,
        tags={},
    )
    db.transaction(lambda session: session.add(event))
    index = LexicalIndex(db)
    index.bulk_upsert([event])
    hits = index.search("hello")
    assert hits
    assert hits[0].event_id == event.event_id
