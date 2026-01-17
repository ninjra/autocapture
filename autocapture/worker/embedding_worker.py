"""Embedding worker for span/event embeddings and vector indexing."""

from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Optional

from sqlalchemy import select, update

from ..config import AppConfig
from ..runtime_governor import RuntimeGovernor, RuntimeMode
from ..embeddings.service import EmbeddingService
from ..indexing.lexical_index import LexicalIndex
from ..indexing.vector_index import IndexUnavailable, SpanEmbeddingUpsert, VectorIndex
from ..indexing.spans_v2 import SpanV2Upsert, SpansV2Index, SparseEmbedding
from ..embeddings.sparse import SparseEncoder
from ..embeddings.late import LateInteractionEncoder
from ..logging_utils import get_logger
from ..observability.metrics import (
    embedding_backlog,
    embedding_latency_ms,
    worker_errors_total,
)
from ..storage.database import DatabaseManager
from ..storage.models import EmbeddingRecord, EventRecord, OCRSpanRecord


class EmbeddingWorker:
    def __init__(
        self,
        config: AppConfig,
        db_manager: DatabaseManager | None = None,
        embedder: Optional[EmbeddingService] = None,
        vector_index: VectorIndex | None = None,
        runtime_governor: RuntimeGovernor | None = None,
    ) -> None:
        self._config = config
        self._db = db_manager or DatabaseManager(config.database)
        self._log = get_logger("worker.embedding")
        self._embedder = embedder or EmbeddingService(config.embed)
        self._vector_index = vector_index or VectorIndex(config, self._embedder.dim)
        self._lexical_index = LexicalIndex(self._db)
        self._runtime = runtime_governor
        self._spans_v2: SpansV2Index | None = None
        self._sparse_encoder: SparseEncoder | None = None
        self._late_encoder: LateInteractionEncoder | None = None
        if (
            config.retrieval.use_spans_v2
            or config.retrieval.sparse_enabled
            or config.retrieval.late_enabled
        ):
            self._spans_v2 = SpansV2Index(config, self._embedder.dim)
            if config.retrieval.sparse_enabled:
                self._sparse_encoder = SparseEncoder(config.retrieval.sparse_model)
            if config.retrieval.late_enabled:
                self._late_encoder = LateInteractionEncoder(
                    dim=int(config.qdrant.late_vector_size), max_tokens=64
                )
        self._lease_timeout_s = config.worker.embedding_lease_ms / 1000
        self._max_attempts = config.worker.embedding_max_attempts
        self._max_task_runtime_s = config.worker.max_task_runtime_s
        self._index_backoff_s = 0.0

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        poll_interval = self._config.worker.poll_interval_s
        self._recover_stale_embeddings()
        self._recover_stale_event_embeddings()
        backoff_s = 1.0
        while True:
            if stop_event and stop_event.is_set():
                return
            if self._runtime and self._runtime.current_mode == RuntimeMode.FULLSCREEN_HARD_PAUSE:
                time.sleep(min(poll_interval, 1.0))
                continue
            try:
                processed = self.process_batch()
                backoff_s = 1.0
            except Exception as exc:
                self._log.exception("Embedding worker loop failed: {}", exc)
                worker_errors_total.labels("embedding").inc()
                if stop_event and stop_event.is_set():
                    return
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 30.0)
                continue
            if processed == 0:
                time.sleep(poll_interval)

    def process_batch(self) -> int:
        self._update_backlog_metrics()
        processed = 0
        processed += self._process_event_embeddings()
        processed += self._process_span_embeddings()
        return processed

    def _update_backlog_metrics(self) -> None:
        try:
            with self._db.session() as session:
                pending_events = (
                    session.execute(
                        select(EventRecord)
                        .where(EventRecord.embedding_status == "pending")
                        .where(EventRecord.embedding_attempts < self._max_attempts)
                    )
                    .scalars()
                    .all()
                )
                pending_spans = (
                    session.execute(
                        select(EmbeddingRecord.id).where(
                            EmbeddingRecord.status.in_(["pending", "index_pending"]),
                            EmbeddingRecord.model == self._embedder.model_name,
                            EmbeddingRecord.attempts < self._max_attempts,
                        )
                    )
                    .scalars()
                    .all()
                )
            embedding_backlog.set(len(pending_events) + len(pending_spans))
        except Exception as exc:
            self._log.debug("Embedding backlog metric failed: {}", exc)

    def _process_event_embeddings(self) -> int:
        batch_size = self._config.embed.text_batch_size
        if self._runtime:
            profile = self._runtime.qos_profile()
            if profile.embed_batch_size:
                batch_size = profile.embed_batch_size
        self._recover_stale_event_embeddings()

        def _claim(
            session,
        ) -> tuple[dt.datetime | None, list[tuple[str, str]]]:
            events = (
                session.execute(
                    select(EventRecord)
                    .where(
                        EventRecord.embedding_status == "pending",
                        EventRecord.embedding_attempts < self._max_attempts,
                    )
                    .order_by(EventRecord.ts_start.desc())
                    .limit(batch_size)
                )
                .scalars()
                .all()
            )
            if not events:
                return None, []

            claimed_at = dt.datetime.now(dt.timezone.utc)
            tasks: list[tuple[str, str]] = []
            for event in events:
                event.embedding_status = "processing"
                event.embedding_started_at = claimed_at
                event.embedding_heartbeat_at = claimed_at
                event.embedding_attempts += 1
                event.embedding_last_error = None
                tasks.append((event.event_id, event.ocr_text or ""))

            return claimed_at, tasks

        claimed_at, tasks = self._db.transaction(_claim)
        if not tasks:
            return 0

        assert claimed_at is not None

        event_ids = [event_id for event_id, _ in tasks]
        texts = [text for _, text in tasks]

        stop_tick = threading.Event()
        tick_interval = max(1.0, min(10.0, self._lease_timeout_s / 3.0))

        def _tick_heartbeat() -> None:
            warned = False
            while not stop_tick.wait(tick_interval):
                if time.monotonic() - start_ts >= self._max_task_runtime_s:
                    if not warned:
                        self._log.warning(
                            "Event embedding heartbeat exceeded max runtime; stopping so lease can be reclaimed."
                        )
                        warned = True
                    return
                now = dt.datetime.now(dt.timezone.utc)
                try:
                    self._db.transaction(
                        lambda session: session.execute(
                            update(EventRecord)
                            .where(EventRecord.event_id.in_(event_ids))
                            .where(
                                EventRecord.embedding_status == "processing",
                                EventRecord.embedding_started_at == claimed_at,
                            )
                            .values(embedding_heartbeat_at=now)
                        )
                    )
                except Exception:
                    return

        ticker = threading.Thread(
            target=_tick_heartbeat,
            name="embedding-event-heartbeat",
            daemon=True,
        )
        start_ts = time.monotonic()
        ticker.start()
        try:
            start = time.monotonic()
            vectors = self._embedder.embed_texts(texts)
            embedding_latency_ms.observe((time.monotonic() - start) * 1000)
        except Exception as exc:
            self._log.warning("Event embedding failed: {}", exc)
            worker_errors_total.labels("embedding").inc()
            error = str(exc)

            self._db.transaction(
                lambda session: session.execute(
                    update(EventRecord)
                    .where(EventRecord.event_id.in_(event_ids))
                    .where(
                        EventRecord.embedding_status == "processing",
                        EventRecord.embedding_started_at == claimed_at,
                    )
                    .values(
                        embedding_status="failed",
                        embedding_last_error=error,
                        embedding_heartbeat_at=dt.datetime.now(dt.timezone.utc),
                    )
                )
            )
            return 0
        finally:
            stop_tick.set()
            ticker.join(timeout=1.0)

        def _persist(session) -> int:
            now = dt.datetime.now(dt.timezone.utc)
            written = 0
            for (event_id, _text), vector in zip(tasks, vectors):
                record = session.get(EventRecord, event_id)
                if not record:
                    continue
                if record.embedding_status != "processing":
                    continue
                if record.embedding_started_at != claimed_at:
                    continue
                record.embedding_vector = vector
                record.embedding_status = "done"
                record.embedding_model = self._embedder.model_name
                record.embedding_last_error = None
                record.embedding_heartbeat_at = now
                written += 1
            return written

        return self._db.transaction(_persist)

    def _process_span_embeddings(self) -> int:
        batch_size = self._config.embed.text_batch_size
        if self._runtime:
            profile = self._runtime.qos_profile()
            if profile.embed_batch_size:
                batch_size = profile.embed_batch_size
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

        self._db.transaction(
            lambda session: session.execute(
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
        )

        stop_tick = threading.Event()
        tick_interval = max(1.0, min(10.0, self._lease_timeout_s / 3.0))
        start_ts = time.monotonic()

        def _tick_span_heartbeat() -> None:
            warned = False
            while not stop_tick.wait(tick_interval):
                if time.monotonic() - start_ts >= self._max_task_runtime_s:
                    if not warned:
                        self._log.warning(
                            "Span embedding heartbeat exceeded max runtime; stopping so lease can be reclaimed."
                        )
                        warned = True
                    return
                now = dt.datetime.now(dt.timezone.utc)
                try:
                    self._db.transaction(
                        lambda session: session.execute(
                            update(EmbeddingRecord)
                            .where(EmbeddingRecord.id.in_(embedding_ids))
                            .where(EmbeddingRecord.status == "processing")
                            .values(heartbeat_at=now)
                        )
                    )
                except Exception:
                    return

        ticker = threading.Thread(
            target=_tick_span_heartbeat,
            name="embedding-span-heartbeat",
            daemon=True,
        )
        ticker.start()

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
                span = (
                    session.execute(
                        select(OCRSpanRecord)
                        .where(OCRSpanRecord.capture_id == embedding.capture_id)
                        .where(OCRSpanRecord.span_key == embedding.span_key)
                    )
                    .scalars()
                    .first()
                )
                event = session.get(EventRecord, embedding.capture_id)
                if not span or not event:
                    continue
                span_rows.append((embedding, span, event))

        if not span_rows:
            stop_tick.set()
            ticker.join(timeout=1.0)
            return 0

        sparse_vectors: dict[str, SparseEmbedding] = {}
        if self._sparse_encoder and self._config.retrieval.sparse_enabled:
            sparse_list = self._sparse_encoder.encode([span.text for _, span, _ in span_rows])
            for (_embedding, span, _event), sparse in zip(span_rows, sparse_list, strict=False):
                sparse_vectors[span.span_key] = sparse

        late_vectors: dict[str, list[list[float]]] = {}
        if self._late_encoder and self._config.retrieval.late_enabled:
            eligible = self._eligible_late_span_keys(span_rows)
            for _embedding, span, _event in span_rows:
                if span.span_key in eligible:
                    late_vectors[span.span_key] = self._late_encoder.encode_text(span.text)

        vectors: list[list[float]] = []
        span_texts = [span.text for embedding, span, _ in span_rows if embedding.vector is None]
        if span_texts:
            try:
                start = time.monotonic()
                vectors = self._embedder.embed_texts(span_texts)
                embedding_latency_ms.observe((time.monotonic() - start) * 1000)
            except Exception as exc:
                self._log.warning("Span embedding failed: {}", exc)
                worker_errors_total.labels("embedding").inc()
                error = str(exc)

                def _mark_failed(session) -> None:
                    session.execute(
                        update(EmbeddingRecord)
                        .where(EmbeddingRecord.id.in_(embedding_ids))
                        .values(status="failed", last_error=error)
                    )

                self._db.transaction(_mark_failed)
                stop_tick.set()
                ticker.join(timeout=1.0)
                return 0

        vectors_iter = iter(vectors)
        upserts: list[SpanEmbeddingUpsert] = []
        spans_v2_upserts: list[SpanV2Upsert] = []

        def _prepare(session) -> None:
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
                    "ts_start": event.ts_start.isoformat(),
                    "app_name": event.app_name,
                    "domain": event.domain or "",
                }
                upserts.append(
                    SpanEmbeddingUpsert(
                        capture_id=event.event_id,
                        span_key=record.span_key,
                        vector=vector,
                        payload=payload,
                        embedding_model=self._embedder.model_name,
                    )
                )
                if self._spans_v2:
                    spans_v2_upserts.append(
                        SpanV2Upsert(
                            capture_id=event.event_id,
                            span_key=record.span_key,
                            dense_vector=vector,
                            sparse_vector=sparse_vectors.get(record.span_key),
                            late_vectors=late_vectors.get(record.span_key),
                            payload=_build_spans_v2_payload(event, span),
                            embedding_model=self._embedder.model_name,
                        )
                    )

        self._db.transaction(_prepare)

        if upserts:
            try:
                self._vector_index.upsert_spans(upserts)
                self._index_backoff_s = 0.0
            except IndexUnavailable as exc:
                self._log.warning("Vector index unavailable: {}", exc)
                worker_errors_total.labels("embedding").inc()
                error = str(exc)
                self._index_backoff_s = min(5.0, self._index_backoff_s * 2 or 0.5)
                time.sleep(self._index_backoff_s)

                def _mark_pending(session) -> None:
                    session.execute(
                        update(EmbeddingRecord)
                        .where(EmbeddingRecord.id.in_(embedding_ids))
                        .values(status="index_pending", last_error=error)
                    )

                self._db.transaction(_mark_pending)
                stop_tick.set()
                ticker.join(timeout=1.0)
                return 0
            except Exception as exc:
                self._log.warning("Span indexing failed: {}", exc)
                worker_errors_total.labels("embedding").inc()
                error = str(exc)

                def _mark_pending(session) -> None:
                    session.execute(
                        update(EmbeddingRecord)
                        .where(EmbeddingRecord.id.in_(embedding_ids))
                        .values(status="index_pending", last_error=error)
                    )

                self._db.transaction(_mark_pending)
                stop_tick.set()
                ticker.join(timeout=1.0)
                return 0

        if spans_v2_upserts and self._spans_v2:
            try:
                self._spans_v2.upsert(spans_v2_upserts)
            except Exception as exc:
                self._log.warning("Spans v2 indexing failed: {}", exc)

        def _finalize(session) -> None:
            for embedding, span, _event in span_rows:
                record = session.get(EmbeddingRecord, embedding.id)
                if not record or record.status == "failed":
                    continue
                record.status = "done"
                record.last_error = None
                record.span_key = record.span_key or span.span_key
                record.updated_at = dt.datetime.now(dt.timezone.utc)

        self._db.transaction(_finalize)
        stop_tick.set()
        ticker.join(timeout=1.0)
        return len(span_rows)

    def _recover_stale_event_embeddings(self) -> None:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=self._lease_timeout_s)

        def _recover(session) -> None:
            rows = (
                session.execute(
                    select(EventRecord).where(EventRecord.embedding_status == "processing")
                )
                .scalars()
                .all()
            )
            for record in rows:
                heartbeat = (
                    record.embedding_heartbeat_at
                    or record.embedding_started_at
                    or record.created_at
                )
                heartbeat = _ensure_aware(heartbeat)
                if heartbeat and heartbeat >= cutoff:
                    continue

                # If the vector is already present, treat this row as completed.
                if record.embedding_vector is not None:
                    record.embedding_status = "done"
                    record.embedding_last_error = None
                    record.embedding_started_at = None
                    record.embedding_heartbeat_at = None
                    continue

                if record.embedding_attempts >= self._max_attempts:
                    record.embedding_status = "failed"
                    record.embedding_last_error = "max_attempts_exceeded"
                else:
                    record.embedding_status = "pending"
                    record.embedding_started_at = None
                    record.embedding_heartbeat_at = None

        self._db.transaction(_recover)

    def _recover_stale_embeddings(self) -> None:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=self._lease_timeout_s)

        def _recover(session) -> None:
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

        self._db.transaction(_recover)

    def _eligible_late_span_keys(
        self, span_rows: list[tuple[EmbeddingRecord, OCRSpanRecord, EventRecord]]
    ) -> set[str]:
        config = self._config.retrieval
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=config.late_max_days)
        per_event: dict[str, list[OCRSpanRecord]] = {}
        for _embedding, span, event in span_rows:
            if event.ts_start < cutoff:
                continue
            if len(span.text or "") > config.late_text_max_chars:
                continue
            per_event.setdefault(event.event_id, []).append(span)
        eligible: set[str] = set()
        for _event_id, spans in per_event.items():
            spans_sorted = sorted(
                spans, key=lambda s: float(getattr(s, "confidence", 0.0)), reverse=True
            )
            for span in spans_sorted[: config.late_max_spans_per_event]:
                eligible.add(span.span_key)
        return eligible


def _ensure_aware(timestamp: dt.datetime | None) -> dt.datetime | None:
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp


def _build_spans_v2_payload(event: EventRecord, span: OCRSpanRecord) -> dict:
    bbox_norm = _normalize_bbox(span.bbox)
    text = (span.text or "").strip()
    if len(text) > 300:
        text = text[:300]
    return {
        "event_id": event.event_id,
        "capture_id": event.event_id,
        "span_id": span.span_key,
        "ts": event.ts_start.isoformat(),
        "app": event.app_name,
        "window_title": event.window_title,
        "domain": event.domain or "",
        "bbox_norm": bbox_norm,
        "tile_id": None,
        "text": text,
        "tags": event.tags or {},
    }


def _normalize_bbox(bbox: object) -> list[float]:
    if isinstance(bbox, list) and bbox:
        values = [float(val) for val in bbox if isinstance(val, (int, float))]
        if not values:
            return []
        max_val = max(values)
        if max_val <= 1.0 and len(values) >= 4:
            return [max(0.0, min(1.0, float(val))) for val in values[:4]]
    return []
