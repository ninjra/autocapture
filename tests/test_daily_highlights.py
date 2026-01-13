from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

from sqlalchemy import select

from autocapture.agents import AGENT_JOB_DAILY_HIGHLIGHTS
from autocapture.agents.jobs import AgentJobQueue
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import AgentJobRecord, DailyHighlightsRecord, EventRecord
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
    def upsert_spans(self, upserts):
        return None


def test_daily_highlights_pipeline() -> None:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    db = DatabaseManager(config.database)
    day = dt.date.today().isoformat()
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="evt-day",
                ts_start=now,
                ts_end=None,
                app_name="Editor",
                window_title="Notes",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="",
                tags={},
            )
        )
    payload = {
        "schema_version": "v1",
        "day": day,
        "summary": "Worked on notes.",
        "highlights": ["Summarized notes"],
        "projects": ["Notes"],
        "open_loops": ["Send summary"],
        "people": [],
        "context_switches": [],
        "time_spent_by_app": {"Editor": 1.5},
        "provenance": {
            "model": "stub",
            "provider": "stub",
            "prompt": "stub",
            "created_at_utc": now.isoformat(),
        },
    }
    queue = AgentJobQueue(db)
    queue.enqueue(
        job_key=f"highlights:{day}:v1",
        job_type=AGENT_JOB_DAILY_HIGHLIGHTS,
        day=day,
        payload={"day": day},
    )
    worker = AgentJobWorker(
        config,
        db_manager=db,
        embedder=_StubEmbedder(),
        vector_index=_StubVectorIndex(),
        llm_client=_StubLLM(payload),
    )
    worker.process_batch()
    with db.session() as session:
        job = session.execute(select(AgentJobRecord).where(AgentJobRecord.day == day)).scalar_one()
        assert job.status == "completed"
        highlight = session.execute(select(DailyHighlightsRecord)).scalars().first()
        assert highlight is not None
