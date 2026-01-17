from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

from sqlalchemy import select

from autocapture.agents import AGENT_JOB_THREAD_SUMMARY
from autocapture.agents.jobs import AgentJobQueue
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.indexing.thread_index import ThreadLexicalIndex
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import (
    EventRecord,
    ThreadEventRecord,
    ThreadRecord,
    ThreadSummaryRecord,
)
from autocapture.worker.agent_worker import AgentJobWorker


@dataclass
class _StubResponse:
    text: str
    model: str = "stub"
    provider: str = "stub"


class _StubLLM:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def generate_text(self, *_args, **_kwargs) -> _StubResponse:
        return _StubResponse(text=json.dumps(self._payload))


class _StubEmbedder:
    model_name = "stub"
    dim = 3

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _StubVectorIndex:
    def __init__(self) -> None:
        self.upserts = []

    def upsert_spans(self, upserts):
        self.upserts.extend(upserts)


def test_thread_summary_job_persists_and_indexes(tmp_path) -> None:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    config.capture.data_dir = tmp_path
    config.capture.staging_dir = tmp_path / "staging"
    db = DatabaseManager(config.database)
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            ThreadRecord(
                thread_id="thread-1",
                ts_start=now,
                ts_end=now + dt.timedelta(minutes=10),
                app_name="Editor",
                window_title="Thread Window",
                event_count=2,
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            EventRecord(
                event_id="evt-1",
                ts_start=now,
                ts_end=None,
                app_name="Editor",
                window_title="Thread Window",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="alpha",
                embedding_vector=None,
                embedding_status="pending",
                embedding_model=None,
                tags={},
            )
        )
        session.flush()
        session.add(ThreadEventRecord(thread_id="thread-1", event_id="evt-1", position=0))

    payload = {
        "schema_version": "v1",
        "thread_id": "thread-1",
        "title": "Project Thread",
        "summary": "Summary text",
        "key_entities": ["Alpha"],
        "tasks": [{"title": "Task A", "status": "done", "evidence": []}],
        "citations": [{"event_id": "evt-1", "ts_start": now.isoformat(), "ts_end": None}],
        "provenance": {
            "model": "stub",
            "provider": "stub",
            "prompt": "stub",
            "created_at_utc": now.isoformat(),
        },
    }
    queue = AgentJobQueue(db)
    queue.enqueue(
        job_key="thread:thread-1:v1",
        job_type=AGENT_JOB_THREAD_SUMMARY,
        event_id="thread-1",
        payload={"thread_id": "thread-1"},
    )
    vector_index = _StubVectorIndex()
    worker = AgentJobWorker(
        config,
        db_manager=db,
        embedder=_StubEmbedder(),
        vector_index=vector_index,
        llm_client=_StubLLM(payload),
    )
    worker.process_batch()

    with db.session() as session:
        summary = session.get(ThreadSummaryRecord, "thread-1")
        assert summary is not None
        assert summary.data_json.get("title") == "Project Thread"
        count = session.execute(select(ThreadSummaryRecord)).scalars().all()
        assert count

    lexical = ThreadLexicalIndex(db)
    hits = lexical.search("Project", limit=5)
    assert hits
    assert hits[0].thread_id == "thread-1"
    assert vector_index.upserts
