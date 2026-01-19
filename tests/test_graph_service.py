from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from autocapture.config import AppConfig, DatabaseConfig, GraphServiceConfig
from autocapture.graph.models import GraphIndexRequest, GraphQueryRequest, GraphTimeRange
from autocapture.graph.service import GraphService
from autocapture.memory.threads import ThreadCandidate
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord, ThreadEventRecord, ThreadRecord


class _StubThreadRetrieval:
    def __init__(self, candidates: list[ThreadCandidate]):
        self._candidates = candidates

    def retrieve(self, _query: str, _time_range, *, limit: int = 5):
        return self._candidates[:limit]


class _StubRetrieval:
    def retrieve(self, *_args, **_kwargs):
        return type("Batch", (), {"results": []})()


def _make_event(event_id: str, ts: dt.datetime) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        ts_start=ts,
        ts_end=None,
        app_name="App",
        window_title="Title",
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


def test_graph_index_writes_manifest(tmp_path: Path) -> None:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False),
        graph_service=GraphServiceConfig(workspace_root=tmp_path, max_events=100),
    )
    db = DatabaseManager(config.database)
    now = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.timezone.utc)
    with db.session() as session:
        session.add(_make_event("e1", now))
    service = GraphService(config, db=db)
    response = service.index(
        GraphIndexRequest(
            corpus_id="default",
            time_range=GraphTimeRange(start=now.isoformat(), end=now.isoformat()),
            max_events=10,
        )
    )
    manifest_path = tmp_path / "default" / "manifest.json"
    assert response.events_indexed == 1
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["events_indexed"] == 1


def test_graph_query_uses_thread_candidates(tmp_path: Path) -> None:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False),
        graph_service=GraphServiceConfig(workspace_root=tmp_path, max_events=100),
    )
    db = DatabaseManager(config.database)
    now = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.timezone.utc)
    with db.session() as session:
        session.add(_make_event("e1", now))
        session.add(
            ThreadRecord(
                thread_id="t1",
                ts_start=now,
                ts_end=now,
                app_name="App",
                window_title="Title",
                event_count=1,
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(ThreadEventRecord(thread_id="t1", event_id="e1", position=0))
    candidate = ThreadCandidate(
        thread_id="t1",
        score=0.9,
        lexical_score=0.9,
        vector_score=0.0,
        title="Title",
        summary="Summary",
        ts_start=now,
        ts_end=None,
        citations=[],
    )
    service = GraphService(
        config,
        db=db,
        thread_retrieval=_StubThreadRetrieval([candidate]),
        retrieval=_StubRetrieval(),
    )
    response = service.query(GraphQueryRequest(corpus_id="default", query="test", limit=5))
    assert response.hits
    assert response.hits[0].event_id == "e1"
