from __future__ import annotations

import datetime as dt

from autocapture.config import DatabaseConfig
from autocapture.indexing.lexical_index import LexicalIndex
from autocapture.indexing.thread_index import ThreadLexicalIndex
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord, ThreadRecord


def _db() -> DatabaseManager:
    cfg = DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False)
    return DatabaseManager(cfg)


def test_lexical_index_fallback_like_search(monkeypatch) -> None:
    monkeypatch.setattr(
        "autocapture.indexing.lexical_index.sqlite_fts5_available",
        lambda _engine: False,
    )
    db = _db()
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="EVT-1",
                ts_start=now,
                ts_end=None,
                app_name="App",
                window_title="Window",
                url=None,
                domain=None,
                screenshot_path=None,
                focus_path=None,
                screenshot_hash="hash",
                frame_hash=None,
                ocr_text="Hello world",
                ocr_text_normalized="hello world",
                tags={},
            )
        )
    hits = LexicalIndex(db).search("hello", limit=5)
    assert any(hit.event_id == "EVT-1" for hit in hits)


def test_thread_index_fallback_like_search(monkeypatch) -> None:
    monkeypatch.setattr(
        "autocapture.indexing.thread_index.sqlite_fts5_available",
        lambda _engine: False,
    )
    db = _db()
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            ThreadRecord(
                thread_id="TH-1",
                ts_start=now,
                ts_end=None,
                app_name="Planner",
                window_title="Weekly Window",
                event_count=1,
            )
        )
    hits = ThreadLexicalIndex(db).search("weekly", limit=5)
    assert any(hit.thread_id == "TH-1" for hit in hits)
