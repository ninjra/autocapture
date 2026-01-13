from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

from sqlalchemy import select

from autocapture.agents import AGENT_JOB_VISION_CAPTION
from autocapture.agents.jobs import AgentJobQueue
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import AgentJobRecord, EventRecord
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


def _config() -> AppConfig:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    config.agents.vision.run_only_when_idle = False
    config.agents.vision.max_jobs_per_hour = 0
    return config


def test_vision_caption_pipeline(tmp_path) -> None:
    config = _config()
    db = DatabaseManager(config.database)
    image_path = tmp_path / "shot.webp"
    image_path.write_bytes(b"fake")
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="evt-vision",
                ts_start=now,
                ts_end=None,
                app_name="Browser",
                window_title="Docs",
                url=None,
                domain=None,
                screenshot_path=str(image_path),
                screenshot_hash="hash",
                ocr_text="",
                tags={},
            )
        )
    payload = {
        "schema_version": "v1",
        "event_id": "evt-vision",
        "caption": "Looking at a docs page.",
        "ui_elements": ["Docs"],
        "visible_text_summary": "Some documentation.",
        "sensitivity": {"contains_pii": False, "contains_secrets": False, "notes": []},
        "provenance": {
            "model": "stub",
            "provider": "stub",
            "prompt": "stub",
            "created_at_utc": now.isoformat(),
        },
    }
    queue = AgentJobQueue(db)
    queue.enqueue(
        job_key="vision:evt-vision:v1",
        job_type=AGENT_JOB_VISION_CAPTION,
        event_id="evt-vision",
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
        job = session.execute(
            select(AgentJobRecord).where(AgentJobRecord.event_id == "evt-vision")
        ).scalar_one()
        assert job.status == "completed"


def test_vision_caption_missing_screenshot() -> None:
    config = _config()
    db = DatabaseManager(config.database)
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="evt-missing",
                ts_start=now,
                ts_end=None,
                app_name="Browser",
                window_title="Docs",
                url=None,
                domain=None,
                screenshot_path="missing.webp",
                screenshot_hash="hash",
                ocr_text="",
                tags={},
            )
        )
    queue = AgentJobQueue(db)
    queue.enqueue(
        job_key="vision:evt-missing:v1",
        job_type=AGENT_JOB_VISION_CAPTION,
        event_id="evt-missing",
    )
    worker = AgentJobWorker(
        config,
        db_manager=db,
        embedder=_StubEmbedder(),
        vector_index=_StubVectorIndex(),
        llm_client=_StubLLM({}),
    )
    worker.process_batch()
    with db.session() as session:
        job = session.execute(
            select(AgentJobRecord).where(AgentJobRecord.event_id == "evt-missing")
        ).scalar_one()
        assert job.status == "skipped"
