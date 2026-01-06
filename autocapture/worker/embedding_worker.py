"""Embedding worker for span/event embeddings and vector indexing."""

from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Optional

from sqlalchemy import select, update

from ..config import AppConfig
from ..embeddings.service import EmbeddingService
from ..indexing.lexical_index import LexicalIndex
from ..indexing.vector_index import VectorIndex
from ..logging_utils import get_logger
from ..observability.metrics import embedding_latency_ms, worker_errors_total
from ..storage.database import DatabaseManager
from ..storage.models import EmbeddingRecord, EventRecord, OCRSpanRecord


class EmbeddingWorker:
    def __init__(
        self,
        config: AppConfig,
        db_manager: DatabaseManager | None = None,
        embedder: Optional[EmbeddingService] = None,
    ) -> None:
        self._config = config
        self._db = db_manager or DatabaseManager(config.database)
        self._log = get_logger("worker.embedding")
        self._embedder = embedder or EmbeddingService(config.embeddings)
        self._vector_index = VectorIndex(config, self._db, self._embedder.dim)
        self._lexical_index = LexicalIndex(self._db)
        self._lease_timeout_s = config.worker.embedding_lease_ms / 1000
        self._max_attempts = config.worker.embedding_max_attempts

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        poll_interval = self._config.worker.poll_interval_s
        self._recover_stale_embeddings()
        while True:
            if stop_event and stop_event.is_set():
                return
            processed = self.process_batch()
            if processed == 0:
                time.sleep(poll_interval)

    def process_batch(self) -> int:
        processed = 0
        processed += self._process_event_embeddings()
        processed += self._process_span_embeddings()
        return processed

    def _process_event_embeddings(self) -> int:
        batch_size = self._config.embeddings.batch_size
        with self._db.session() as session:
            event_ids = (
                session.execute(
                    select(EventRecord.event_id)
                    .where(EventRecord.embedding_status == "pending")
                    .limit(batch_size)
                )
                .scalars()
                .all()
            )
            if not event_ids:
                return 0
            session.execute(
                update(EventRecord)
                .where(EventRecord.event_id.in_(event_ids))
                .where(EventRecord.embedding_status == "pending")
                .values(embedding_status="processing")
            )

        with self._db.session() as session:
            events = (
                session.execute(select(EventRecord).where(EventRecord.event_id.in_(event_ids)))
                .scalars()
                .all()
            )
            texts = [event.ocr_text or "" for event in events]
        try:
            start = time.monotonic()
            vectors = self._embedder.embed_texts(texts)
            embedding_latency_ms.observe((time.monotonic() - start) * 1000)
        except Exception as exc:
            self._log.warning("Event embedding failed: {}", exc)
            worker_errors_total.labels("embedding").inc()
            with self._db.session() as session:
                session.execute(
                    update(EventRecord)
                    .where(EventRecord.event_id.in_(event_ids))
                    .values(embedding_status="failed")
                )
            return 0
        with self._db.session() as session:
            for event, vector in zip(events, vectors):
                record = session.get(EventRecord, event.event_id)
                if not record:
                    continue
                record.embedding_vector = vector
                record.embedding_status = "done"
                record.embedding_model = self._embedder.model_name
        return len(events)

    def _process_span_embeddings(self) -> int:
        batch_size = self._config.embeddings.batch_size
        self._recover_stale_embeddings()
        with self._db.session() as session:
            embedding_ids = (
                session.execute(
                    select(EmbeddingRecord.id)
                    .where(
                        EmbeddingRecord.status.in_(["pending", "index_pending"]),
                        EmbeddingRecord.model == self._embedder.model_name,
                        EmbeddingRecord.attempts < self._max_attempts,
                    )
                    .limit(batch_size)
                )
                .scalars()
                .all()
            )
            if not embedding_ids:
                return 0
            session.execute(
                update(EmbeddingRecord)
                .where(EmbeddingRecord.id.in_(embedding_ids))
                .where(
                    EmbeddingRecord.status.in_(["pending", "index_pending"]),
                    EmbeddingRecord.attempts < self._max_attempts,
                )
                .values(
                    status="processing",
                    processing_started_at=dt.datetime.now(dt.timezone.utc),
                    heartbeat_at=dt.datetime.now(dt.timezone.utc),
                    attempts=EmbeddingRecord.attempts + 1,
                )
            )

        with self._db.session() as session:
            embeddings = (
                session.execute(
                    select(EmbeddingRecord).where(EmbeddingRecord.id.in_(embedding_ids))
                )
                .scalars()
                .all()
            )
            span_rows = []
            for embedding in embeddings:
                span = session.get(OCRSpanRecord, embedding.span_id)
                event = session.get(EventRecord, embedding.capture_id)
                if not span or not event:
                    continue
                span_rows.append((embedding, span, event))

        if not span_rows:
            return 0

        vectors: list[list[float]] = []
        span_texts = [
            span.text for embedding, span, _ in span_rows if embedding.vector is None
        ]
        if span_texts:
            try:
                start = time.monotonic()
                vectors = self._embedder.embed_texts(span_texts)
                embedding_latency_ms.observe((time.monotonic() - start) * 1000)
            except Exception as exc:
                self._log.warning("Span embedding failed: {}", exc)
                worker_errors_total.labels("embedding").inc()
                with self._db.session() as session:
                    session.execute(
                        update(EmbeddingRecord)
                        .where(EmbeddingRecord.id.in_(embedding_ids))
                        .values(status="failed", last_error=str(exc))
                    )
                return 0

        vectors_iter = iter(vectors)
        upserts = []
        with self._db.session() as session:
            for embedding, span, event in span_rows:
                record = session.get(EmbeddingRecord, embedding.id)
                if not record:
                    continue
                if record.attempts > self._max_attempts:
                    record.status = "failed"
                    record.last_error = "max_attempts_exceeded"
                    continue
                if record.vector is None:
                    vector = next(vectors_iter)
                    record.vector = vector
                else:
                    vector = record.vector
                record.status = "index_pending"
                record.last_error = None
                record.span_key = record.span_key or span.span_key
                record.heartbeat_at = dt.datetime.now(dt.timezone.utc)
                record.updated_at = dt.datetime.now(dt.timezone.utc)
                payload = {
                    "event_id": event.event_id,
                    "span_key": record.span_key,
                    "span_id": span.id,
                    "ts_start": event.ts_start.isoformat(),
                    "app_name": event.app_name,
                    "domain": event.domain or "",
                }
                upserts.append((event.event_id, record.span_key, vector, payload))

        if upserts:
            try:
                self._vector_index.upsert_spans(upserts)
            except Exception as exc:
                self._log.warning("Span indexing failed: {}", exc)
                worker_errors_total.labels("embedding").inc()
                with self._db.session() as session:
                    session.execute(
                        update(EmbeddingRecord)
                        .where(EmbeddingRecord.id.in_(embedding_ids))
                        .values(status="index_pending", last_error=str(exc))
                    )
                return 0

        with self._db.session() as session:
            for embedding, span, _event in span_rows:
                record = session.get(EmbeddingRecord, embedding.id)
                if not record or record.status == "failed":
                    continue
                record.status = "done"
                record.last_error = None
                record.span_key = record.span_key or span.span_key
                record.updated_at = dt.datetime.now(dt.timezone.utc)
        return len(span_rows)

    def _recover_stale_embeddings(self) -> None:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            seconds=self._lease_timeout_s
        )
        with self._db.session() as session:
            rows = (
                session.execute(
                    select(EmbeddingRecord).where(EmbeddingRecord.status == "processing")
                )
                .scalars()
                .all()
            )
            for record in rows:
                heartbeat = record.heartbeat_at or record.processing_started_at or record.updated_at
                heartbeat = _ensure_aware(heartbeat)
                if heartbeat and heartbeat >= cutoff:
                    continue
                if record.attempts >= self._max_attempts:
                    record.status = "failed"
                    record.last_error = "max_attempts_exceeded"
                else:
                    record.status = "index_pending" if record.vector else "pending"
                    record.processing_started_at = None
                    record.heartbeat_at = None


def _ensure_aware(timestamp: dt.datetime | None) -> dt.datetime | None:
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp
