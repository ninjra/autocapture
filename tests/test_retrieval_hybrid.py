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

    def search(self, vector, limit=20, *, filters=None, embedding_model: str | None = None):
        return self._hits


class StubReranker:
    def rank(self, query, documents):
        return [1.0 if "Slack" in doc else 0.1 for doc in documents]


def test_hybrid_retrieval_prioritizes_relevant_event(tmp_path: Path) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
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
                bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
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
                bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            )
        )

    retrieval = RetrievalService(db, config)
    retrieval._embedder = FakeEmbedder()  # type: ignore[attr-defined]
    retrieval._lexical = FakeLexical([LexicalHit(event_id="EOLD", score=0.9)])  # type: ignore[attr-defined]
    retrieval._vector = FakeVector(
        [VectorHit(event_id="EOLD", span_key="S1", score=0.9)]
    )  # type: ignore[attr-defined]

    batch = retrieval.retrieve("roadmap", None, None, limit=2)
    results = batch.results
    assert results
    assert results[0].event.event_id == "EOLD"


def test_reranker_overrides_hybrid_scores(tmp_path: Path) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.embed.text_model = "local-test"
    config.routing.reranker = "enabled"
    config.reranker.enabled = True
    db = DatabaseManager(config.database)

    event_a = EventRecord(
        event_id="EOLD",
        ts_start=dt.datetime.now(dt.timezone.utc),
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
    event_b = EventRecord(
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
        session.add(event_a)
        session.add(event_b)

    retrieval = RetrievalService(db, config, reranker=StubReranker())
    retrieval._embedder = FakeEmbedder()  # type: ignore[attr-defined]
    retrieval._lexical = FakeLexical(
        [
            LexicalHit(event_id="EOLD", score=0.9),
            LexicalHit(event_id="ENEW", score=0.2),
        ]
    )  # type: ignore[attr-defined]
    retrieval._vector = FakeVector(
        [
            VectorHit(event_id="EOLD", span_key="S1", score=0.9),
            VectorHit(event_id="ENEW", span_key="S1", score=0.1),
        ]
    )  # type: ignore[attr-defined]

    batch = retrieval.retrieve("roadmap", None, None, limit=2)
    results = batch.results
    assert results
    assert results[0].event.event_id == "ENEW"


def test_retrieve_offset_applies_to_fallback_query(tmp_path: Path) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.embed.text_model = "local-test"
    db = DatabaseManager(config.database)

    now = dt.datetime.now(dt.timezone.utc)
    events = [
        EventRecord(
            event_id="E1",
            ts_start=now - dt.timedelta(hours=2),
            ts_end=None,
            app_name="Notion",
            window_title="Doc",
            url=None,
            domain=None,
            screenshot_path=None,
            screenshot_hash="hash1",
            ocr_text="alpha beta",
            embedding_vector=None,
            tags={},
        ),
        EventRecord(
            event_id="E2",
            ts_start=now - dt.timedelta(hours=1),
            ts_end=None,
            app_name="Notion",
            window_title="Doc",
            url=None,
            domain=None,
            screenshot_path=None,
            screenshot_hash="hash2",
            ocr_text="alpha gamma",
            embedding_vector=None,
            tags={},
        ),
        EventRecord(
            event_id="E3",
            ts_start=now,
            ts_end=None,
            app_name="Notion",
            window_title="Doc",
            url=None,
            domain=None,
            screenshot_path=None,
            screenshot_hash="hash3",
            ocr_text="alpha delta",
            embedding_vector=None,
            tags={},
        ),
    ]
    with db.session() as session:
        session.add_all(events)

    retrieval = RetrievalService(db, config)
    retrieval._embedder = FakeEmbedder()  # type: ignore[attr-defined]
    retrieval._lexical = FakeLexical([])  # type: ignore[attr-defined]
    retrieval._vector = FakeVector([])  # type: ignore[attr-defined]

    batch = retrieval.retrieve("alpha", None, None, limit=1, offset=1)
    results = batch.results
    assert results
    assert results[0].event.event_id == "E2"
