from __future__ import annotations

import datetime as dt

from sqlalchemy import select

from autocapture.config import AppConfig, DatabaseConfig, EmbeddingConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import (
    CaptureRecord,
    EmbeddingRecord,
    EventRecord,
    OCRSpanRecord,
)
from autocapture.worker.embedding_worker import EmbeddingWorker


class FakeEmbedder:
    def __init__(self) -> None:
        self.model_name = "fake-model"
        self.dim = 3

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeIndex:
    def __init__(self) -> None:
        self.fail = True

    def upsert_spans(self, items):
        if self.fail:
            raise RuntimeError("index down")


def test_embedding_worker_retries_indexing(tmp_path) -> None:
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
        embeddings=EmbeddingConfig(model="fake-model"),
    )
    db = DatabaseManager(config.database)

    with db.session() as session:
        session.add(
            CaptureRecord(
                id="cap-1",
                captured_at=dt.datetime.now(dt.timezone.utc),
                image_path=None,
                foreground_process="test",
                foreground_window="test",
                monitor_id="1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.add(
            EventRecord(
                event_id="cap-1",
                ts_start=dt.datetime.now(dt.timezone.utc),
                ts_end=None,
                app_name="test",
                window_title="test",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="hello",
                embedding_vector=None,
                embedding_status="done",
                embedding_model="fake-model",
                tags={},
            )
        )
        span = OCRSpanRecord(
            capture_id="cap-1",
            span_key="S1",
            start=0,
            end=5,
            text="hello",
            confidence=0.9,
            bbox={},
        )
        session.add(span)
        session.add(
            EmbeddingRecord(
                capture_id="cap-1",
                vector=None,
                model="fake-model",
                status="pending",
                span_key="S1",
            )
        )

    worker = EmbeddingWorker(config, db_manager=db, embedder=FakeEmbedder())
    fake_index = FakeIndex()
    worker._vector_index = fake_index

    processed = worker.process_batch()
    assert processed == 0

    with db.session() as session:
        record = session.execute(
            select(EmbeddingRecord).where(EmbeddingRecord.capture_id == "cap-1")
        ).scalar_one()
        assert record.status == "index_pending"

    fake_index.fail = False
    worker.process_batch()

    with db.session() as session:
        record = session.execute(
            select(EmbeddingRecord).where(EmbeddingRecord.capture_id == "cap-1")
        ).scalar_one()
        assert record.status == "done"


def test_embedding_worker_reclaims_stale_processing(tmp_path) -> None:
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
        embeddings=EmbeddingConfig(model="fake-model"),
    )
    db = DatabaseManager(config.database)

    with db.session() as session:
        session.add(
            CaptureRecord(
                id="cap-2",
                captured_at=dt.datetime.now(dt.timezone.utc),
                image_path=None,
                foreground_process="test",
                foreground_window="test",
                monitor_id="1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.add(
            EventRecord(
                event_id="cap-2",
                ts_start=dt.datetime.now(dt.timezone.utc),
                ts_end=None,
                app_name="test",
                window_title="test",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="hello",
                embedding_vector=None,
                embedding_status="done",
                embedding_model="fake-model",
                tags={},
            )
        )
        span = OCRSpanRecord(
            capture_id="cap-2",
            span_key="S1",
            start=0,
            end=5,
            text="hello",
            confidence=0.9,
            bbox={},
        )
        session.add(span)
        session.add(
            EmbeddingRecord(
                capture_id="cap-2",
                vector=None,
                model="fake-model",
                status="processing",
                span_key="S1",
                heartbeat_at=dt.datetime.now(dt.timezone.utc)
                - dt.timedelta(minutes=10),
            )
        )

    worker = EmbeddingWorker(config, db_manager=db, embedder=FakeEmbedder())
    worker._vector_index = FakeIndex()
    worker._vector_index.fail = False
    worker.process_batch()

    with db.session() as session:
        record = session.execute(
            select(EmbeddingRecord).where(EmbeddingRecord.capture_id == "cap-2")
        ).scalar_one()
        assert record.status == "done"
