from __future__ import annotations

import datetime as dt

from autocapture.config import DatabaseConfig
from autocapture.indexing.lexical_index import LexicalIndex
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord


def test_lexical_index_sanitizes_invalid_queries() -> None:
    db = DatabaseManager(DatabaseConfig(url="sqlite:///:memory:"))
    index = LexicalIndex(db)

    event = EventRecord(
        event_id="evt-1",
        ts_start=dt.datetime.now(dt.timezone.utc),
        ts_end=None,
        app_name="TestApp",
        window_title="Hello World",
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash",
        ocr_text='email "test@example.com" path C:\\Users\\Test',
        embedding_vector=None,
        embedding_status="pending",
        embedding_model=None,
        tags={},
    )
    index.upsert_event(event)

    results = index.search('foo:"bar" baz-qux')
    assert isinstance(results, list)
