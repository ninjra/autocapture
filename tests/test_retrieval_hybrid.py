from __future__ import annotations

import datetime as dt
from pathlib import Path

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.memory.retrieval import RetrievalService
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord, OCRSpanRecord
from autocapture.indexing.lexical_index import LexicalHit
from autocapture.indexing.vector_index import VectorHit


class FakeEmbedder:
    dim = 3

    def embed_texts(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    @property
    def model_name(self):
        return "fake"


class FakeLexical:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, limit=20):
        return self._hits


class FakeVector:
    def __init__(self, hits):
        self._hits = hits

    def search(
        self, vector, limit=20, *, filters=None, embedding_model: str | None = None
    ):
        return self._hits


def test_hybrid_retrieval_prioritizes_relevant_event(tmp_path: Path) -> None:
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}")
    )
    config.embed.text_model = "local-test"
    db = DatabaseManager(config.database)

    old_event = EventRecord(
        event_id="EOLD",
        ts_start=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=5),
        ts_end=None,
        app_name="Notion",
        window_title="Roadmap",
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash",
        ocr_text="Project roadmap details",
        embedding_vector=None,
        tags={},
    )
    new_event = EventRecord(
        event_id="ENEW",
        ts_start=dt.datetime.now(dt.timezone.utc),
        ts_end=None,
        app_name="Slack",
        window_title="Chat",
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash2",
        ocr_text="Random chatter",
        embedding_vector=None,
        tags={},
    )
    with db.session() as session:
        session.add(
            CaptureRecord(
                id="EOLD",
                captured_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=5),
                image_path=None,
                foreground_process="Notion",
                foreground_window="Roadmap",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.add(
            CaptureRecord(
                id="ENEW",
                captured_at=dt.datetime.now(dt.timezone.utc),
                image_path=None,
                foreground_process="Slack",
                foreground_window="Chat",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.flush()
        session.add(old_event)
        session.add(new_event)
        session.add(
            OCRSpanRecord(
                capture_id="EOLD",
                span_key="S1",
                start=8,
                end=15,
                text="roadmap",
                confidence=0.9,
                bbox={},
            )
        )
        session.add(
            OCRSpanRecord(
                capture_id="ENEW",
                span_key="S1",
                start=0,
                end=6,
                text="Random",
                confidence=0.9,
                bbox={},
            )
        )

    retrieval = RetrievalService(db, config)
    retrieval._embedder = FakeEmbedder()  # type: ignore[attr-defined]
    retrieval._lexical = FakeLexical([LexicalHit(event_id="EOLD", score=0.9)])  # type: ignore[attr-defined]
    retrieval._vector = FakeVector(
        [VectorHit(event_id="EOLD", span_key="S1", score=0.9)]
    )  # type: ignore[attr-defined]

    results = retrieval.retrieve("roadmap", None, None, limit=2)
    assert results
    assert results[0].event.event_id == "EOLD"
