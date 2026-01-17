from __future__ import annotations

import datetime as dt

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.indexing.thread_index import ThreadLexicalIndex
from autocapture.memory.threads import ThreadSegmenter, ThreadRetrievalService
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord, ThreadRecord, ThreadSummaryRecord


class _StubEmbedder:
    model_name = "stub"
    dim = 3

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _StubVectorIndex:
    def search(self, *_args, **_kwargs):
        return []

    def upsert_spans(self, *_args, **_kwargs):
        return None


def _make_event(event_id: str, ts: dt.datetime, app: str, title: str) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        ts_start=ts,
        ts_end=None,
        app_name=app,
        window_title=title,
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash",
        ocr_text="text",
        embedding_vector=None,
        embedding_status="pending",
        embedding_model=None,
        tags={},
    )


def test_thread_segmentation_deterministic() -> None:
    segmenter = ThreadSegmenter(max_gap_minutes=30, app_similarity=0.5, title_similarity=0.5)
    base = dt.datetime(2026, 1, 15, 10, 0, tzinfo=dt.timezone.utc)
    events = [
        _make_event("e1", base, "Chrome", "Docs"),
        _make_event("e2", base + dt.timedelta(minutes=5), "Chrome", "Docs"),
        _make_event("e3", base + dt.timedelta(minutes=70), "Terminal", "bash"),
    ]
    segments = segmenter.segment(events)
    assert len(segments) == 2
    assert segments[0].event_ids == ["e1", "e2"]
    assert segments[1].event_ids == ["e3"]


def test_thread_summary_retrieval() -> None:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    db = DatabaseManager(config.database)
    now = dt.datetime(2026, 1, 15, 9, 0, tzinfo=dt.timezone.utc)
    with db.session() as session:
        session.add(
            ThreadRecord(
                thread_id="thread-1",
                ts_start=now,
                ts_end=now + dt.timedelta(minutes=10),
                app_name="Editor",
                window_title="Project Alpha",
                event_count=2,
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            ThreadSummaryRecord(
                thread_id="thread-1",
                schema_version="v1",
                data_json={
                    "thread_id": "thread-1",
                    "title": "Project Alpha",
                    "summary": "Worked on SQL migration plan",
                    "key_entities": ["Alpha"],
                    "tasks": [{"title": "Plan migration", "status": "done", "evidence": []}],
                    "citations": [{"event_id": "e1", "ts_start": now.isoformat(), "ts_end": None}],
                    "provenance": {"model": "stub", "provider": "stub", "prompt": "stub"},
                },
                provenance={"model": "stub", "provider": "stub", "prompt": "stub"},
                created_at=now,
                updated_at=now,
            )
        )

    thread_lexical = ThreadLexicalIndex(db)
    thread_lexical.upsert_thread(
        thread_id="thread-1",
        title="Project Alpha",
        summary="Worked on SQL migration plan",
        entities=["Alpha"],
        tasks=["Plan migration"],
    )
    retrieval = ThreadRetrievalService(
        config,
        db,
        embedder=_StubEmbedder(),
        vector_index=_StubVectorIndex(),
        lexical_index=thread_lexical,
    )
    results = retrieval.retrieve("migration", None, limit=3)
    assert results
    assert results[0].thread_id == "thread-1"
