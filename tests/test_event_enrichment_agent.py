from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

from sqlalchemy import select

from autocapture.agents import AGENT_JOB_ENRICH_EVENT
from autocapture.agents.jobs import AgentJobQueue
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import AgentJobRecord, AgentResultRecord, EventRecord
from autocapture.worker.agent_worker import AgentJobWorker


@dataclass
class _StubResponse:
    text: str


class _StubLLM:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def generate_text(self, *_args, **_kwargs) -> _StubResponse:
        return _StubResponse(text=json.dumps(self._payload))

    def generate_vision(self, *_args, **_kwargs) -> _StubResponse:
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


def test_event_enrichment_pipeline() -> None:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    db = DatabaseManager(config.database)
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="evt-1",
                ts_start=now,
                ts_end=None,
                app_name="Notes",
                window_title="Daily notes",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="Meeting notes about project Apollo",
                tags={},
            )
        )
    payload = {
        "schema_version": "v1",
        "event_id": "evt-1",
        "short_summary": "Summarized meeting notes.",
        "what_i_was_doing": "Writing a summary.",
        "apps_and_tools": ["Notes"],
        "topics": ["Apollo"],
        "tasks": [{"title": "Summarize notes", "status": "done", "evidence": []}],
        "people": [],
        "projects": ["Apollo"],
        "next_actions": ["Send recap"],
        "importance": 0.5,
        "sensitivity": {"contains_pii": False, "contains_secrets": False, "notes": []},
        "keywords": ["meeting"],
        "provenance": {
            "model": "stub",
            "provider": "stub",
            "prompt": "stub",
            "created_at_utc": now.isoformat(),
        },
    }
    queue = AgentJobQueue(db)
    queue.enqueue(job_key="enrich:evt-1:v1", job_type=AGENT_JOB_ENRICH_EVENT, event_id="evt-1")
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
        job = session.execute(
            select(AgentJobRecord).where(AgentJobRecord.event_id == "evt-1")
        ).scalar_one()
        assert job.status == "completed"
        result = session.execute(select(AgentResultRecord)).scalars().first()
        assert result is not None
        event = session.get(EventRecord, "evt-1")
        assert event is not None
        assert event.tags["agents"]["enrichment"]["v1"]["short_summary"]
    assert vector_index.upserts
