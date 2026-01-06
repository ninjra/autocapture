"""Event ingest worker for OCR and event creation."""

from __future__ import annotations

import datetime as dt
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

from ..config import AppConfig
from ..image_utils import ensure_rgb, hash_rgb_image
from ..indexing.lexical_index import LexicalIndex
from ..logging_utils import get_logger
from ..media.store import MediaStore
from ..observability.metrics import ocr_backlog, ocr_latency_ms, worker_errors_total
from ..storage.database import DatabaseManager
from ..storage.models import CaptureRecord, EmbeddingRecord, EventRecord, OCRSpanRecord


@dataclass(frozen=True)
class CapturePayload:
    capture_id: str
    captured_at: dt.datetime
    image_path: str | None
    foreground_process: str
    foreground_window: str
    monitor_id: str
    is_fullscreen: bool


class OCRProcessor:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        self._warmup()

    def _warmup(self) -> None:
        sample = np.zeros((16, 16, 3), dtype=np.uint8)
        self._engine(sample)

    def run(self, image: np.ndarray) -> list[tuple[str, float, list[int]]]:
        bgr = image[:, :, ::-1]
        results, _ = self._engine(bgr)
        spans = []
        for result in results or []:
            box, text, confidence = result
            flattened = [int(coord) for point in box for coord in point]
            spans.append((text, float(confidence), flattened))
        return spans


class EventIngestWorker:
    def __init__(
        self,
        config: AppConfig,
        db_manager: DatabaseManager | None = None,
        ocr_processor: Optional[object] = None,
    ) -> None:
        self._config = config
        self._db = db_manager or DatabaseManager(config.database)
        self._log = get_logger("worker.event_ingest")
        self._lexical = LexicalIndex(self._db)
        self._media_store = MediaStore(config.capture, config.encryption)
        self._lease_timeout_s = config.worker.ocr_lease_ms / 1000
        self._max_attempts = config.worker.ocr_max_attempts
        if ocr_processor is None:
            self._ocr = OCRProcessor()
        else:
            self._ocr = ocr_processor

    def process_batch(self, limit: Optional[int] = None) -> int:
        processed = 0
        self._recover_stale_captures()
        if limit is None:
            limit = self._config.ocr.batch_size
        with self._db.session() as session:
            capture_ids = (
                session.execute(
                    select(CaptureRecord.id)
                    .where(CaptureRecord.ocr_status == "pending")
                    .where(CaptureRecord.ocr_attempts < self._max_attempts)
                    .order_by(CaptureRecord.captured_at.asc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
        ocr_backlog.set(len(capture_ids))

        for capture_id in capture_ids:
            claimed = self._claim_capture(capture_id)
            if not claimed:
                continue
            try:
                start = time.monotonic()
                did_work = self._ingest_capture(capture_id)
                if did_work:
                    ocr_latency_ms.observe((time.monotonic() - start) * 1000)
            except Exception as exc:
                self._log.exception("Failed to ingest capture {}: {}", capture_id, exc)
                worker_errors_total.labels("ocr").inc()
                self._mark_failed(capture_id, str(exc))
                continue
            processed += 1
        return processed

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        poll_interval = self._config.worker.poll_interval_s
        self._recover_stale_captures()
        while True:
            if stop_event and stop_event.is_set():
                return
            processed = self.process_batch()
            if processed == 0:
                time.sleep(poll_interval)

    def _claim_capture(self, capture_id: str) -> bool:
        now = dt.datetime.now(dt.timezone.utc)

        def _claim(session) -> bool:
            result = session.execute(
                update(CaptureRecord)
                .where(
                    CaptureRecord.id == capture_id,
                    CaptureRecord.ocr_status == "pending",
                    CaptureRecord.ocr_attempts < self._max_attempts,
                )
                .values(
                    ocr_status="processing",
                    ocr_started_at=now,
                    ocr_heartbeat_at=now,
                    ocr_attempts=CaptureRecord.ocr_attempts + 1,
                )
            )
            return result.rowcount == 1

        return bool(self._db.transaction(_claim))

    def _mark_failed(self, capture_id: str, error: str | None = None) -> None:
        def _fail(session) -> None:
            capture = session.get(CaptureRecord, capture_id)
            if capture:
                capture.ocr_status = "failed"
                capture.ocr_last_error = error

        self._db.transaction(_fail)

    def _ingest_capture(self, capture_id: str) -> bool:
        capture = self._load_capture(capture_id)
        if not capture:
            return False

        existing_event = self._load_event(capture_id)
        if existing_event and existing_event.ocr_spans:
            self._persist_ocr_results(
                capture,
                existing_event.ocr_text or "",
                existing_event.ocr_spans,
                event_existing=True,
            )
            self._lexical.upsert_event(existing_event)
            return True

        if not capture.image_path:
            self._mark_failed(capture_id, "missing_image_path")
            return False

        path = Path(capture.image_path)
        if not path.exists():
            self._mark_failed(capture_id, "missing_media")
            return False

        stop_event = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(capture_id, stop_event),
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            image = self._load_image(path)
            spans = self._ocr.run(image)
            ocr_text, ocr_spans = _build_ocr_payload(spans)
            self._persist_ocr_results(
                capture,
                ocr_text,
                ocr_spans,
                event_existing=bool(existing_event),
                screenshot_hash=hash_rgb_image(image),
            )
            event = self._load_event(capture_id)
            if event:
                self._lexical.upsert_event(event)
        except FileNotFoundError:
            self._mark_failed(capture_id, "missing_media")
            return False
        except Exception:
            self._mark_failed(capture_id, "corrupt_image")
            return False
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=1.0)
        return True

    def _load_capture(self, capture_id: str) -> CapturePayload | None:
        with self._db.session() as session:
            capture = session.get(CaptureRecord, capture_id)
            if not capture:
                return None
            return CapturePayload(
                capture_id=capture.id,
                captured_at=capture.captured_at,
                image_path=capture.image_path,
                foreground_process=capture.foreground_process,
                foreground_window=capture.foreground_window,
                monitor_id=capture.monitor_id,
                is_fullscreen=capture.is_fullscreen,
            )

    def _load_event(self, event_id: str) -> EventRecord | None:
        with self._db.session() as session:
            return session.get(EventRecord, event_id)

    def _heartbeat_loop(self, capture_id: str, stop_event: threading.Event) -> None:
        interval = max(self._lease_timeout_s / 3, 1.0)
        while not stop_event.wait(interval):
            now = dt.datetime.now(dt.timezone.utc)

            def _tick(session) -> None:
                session.execute(
                    update(CaptureRecord)
                    .where(CaptureRecord.id == capture_id)
                    .values(ocr_heartbeat_at=now)
                )

            try:
                self._db.transaction(_tick)
            except Exception:
                return

    def _persist_ocr_results(
        self,
        capture: CapturePayload,
        ocr_text: str,
        ocr_spans: list[dict],
        *,
        event_existing: bool,
        screenshot_hash: str | None = None,
    ) -> None:
        def _write(session) -> None:
            if not event_existing:
                event = EventRecord(
                    event_id=capture.capture_id,
                    ts_start=capture.captured_at,
                    ts_end=None,
                    app_name=capture.foreground_process,
                    window_title=capture.foreground_window,
                    url=None,
                    domain=_extract_domain(capture.foreground_window, ocr_text),
                    screenshot_path=capture.image_path,
                    screenshot_hash=screenshot_hash or "",
                    ocr_text=ocr_text,
                    ocr_spans=ocr_spans,
                    embedding_vector=None,
                    embedding_status="pending",
                    embedding_model=self._config.embeddings.model,
                    tags={},
                )
                session.add(event)
                if session.bind and session.bind.dialect.name == "sqlite":
                    session.flush()
            else:
                event = session.get(EventRecord, capture.capture_id)
                if event and not event.ocr_spans:
                    event.ocr_text = ocr_text
                    event.ocr_spans = ocr_spans
                    if screenshot_hash and not event.screenshot_hash:
                        event.screenshot_hash = screenshot_hash
            if ocr_spans:
                self._upsert_spans(session, capture.capture_id, ocr_spans)
                span_map = {
                    span.span_key: span
                    for span in session.execute(
                        select(OCRSpanRecord).where(
                            OCRSpanRecord.capture_id == capture.capture_id,
                            OCRSpanRecord.span_key.in_(
                                [str(span.get("span_key")) for span in ocr_spans]
                            ),
                        )
                    )
                    .scalars()
                    .all()
                }
                self._upsert_embeddings(session, capture.capture_id, span_map)
            record = session.get(CaptureRecord, capture.capture_id)
            if record:
                record.ocr_status = "done"
                record.ocr_last_error = None

        self._db.transaction(_write)

    def _upsert_spans(self, session, capture_id: str, ocr_spans: list[dict]) -> None:
        rows = [
            {
                "capture_id": capture_id,
                "span_key": str(span.get("span_key")),
                "start": int(span.get("start", 0)),
                "end": int(span.get("end", 0)),
                "text": str(span.get("text", "")),
                "confidence": float(span.get("conf", 0.0)),
                "bbox": span.get("bbox", []),
            }
            for span in ocr_spans
        ]
        if not rows:
            return
        dialect = session.bind.dialect.name if session.bind else ""
        if dialect == "sqlite":
            stmt = (
                sqlite_insert(OCRSpanRecord)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["capture_id", "span_key"])
            )
            session.execute(stmt)
        elif dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = (
                pg_insert(OCRSpanRecord)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["capture_id", "span_key"])
            )
            session.execute(stmt)
        else:
            for row in rows:
                with session.begin_nested():
                    session.add(OCRSpanRecord(**row))
                    try:
                        session.flush()
                    except IntegrityError:
                        session.rollback()

    def _upsert_embeddings(
        self, session, capture_id: str, span_map: dict[str, OCRSpanRecord]
    ) -> None:
        rows = []
        for span_key, span in span_map.items():
            rows.append(
                {
                    "capture_id": capture_id,
                    "span_id": span.id,
                    "vector": None,
                    "model": self._config.embeddings.model,
                    "status": "pending",
                    "span_key": span_key,
                }
            )
        if not rows:
            return
        dialect = session.bind.dialect.name if session.bind else ""
        if dialect == "sqlite":
            stmt = (
                sqlite_insert(EmbeddingRecord)
                .values(rows)
                .on_conflict_do_nothing(
                    index_elements=["capture_id", "span_key", "model"]
                )
            )
            session.execute(stmt)
        elif dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = (
                pg_insert(EmbeddingRecord)
                .values(rows)
                .on_conflict_do_nothing(
                    index_elements=["capture_id", "span_key", "model"]
                )
            )
            session.execute(stmt)
        else:
            for row in rows:
                with session.begin_nested():
                    session.add(EmbeddingRecord(**row))
                    try:
                        session.flush()
                    except IntegrityError:
                        session.rollback()

    def _load_image(self, path: Path) -> np.ndarray:
        return ensure_rgb(self._media_store.read_image(path))

    def _recover_stale_captures(self) -> None:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            seconds=self._lease_timeout_s
        )

        def _recover(session) -> None:
            rows = (
                session.execute(
                    select(CaptureRecord).where(
                        CaptureRecord.ocr_status == "processing"
                    )
                )
                .scalars()
                .all()
            )
            for capture in rows:
                heartbeat = (
                    capture.ocr_heartbeat_at
                    or capture.ocr_started_at
                    or capture.captured_at
                )
                heartbeat = _ensure_aware(heartbeat)
                if heartbeat and heartbeat >= cutoff:
                    continue
                if capture.ocr_attempts >= self._max_attempts:
                    capture.ocr_status = "failed"
                    capture.ocr_last_error = "max_attempts_exceeded"
                else:
                    capture.ocr_status = "pending"
                    capture.ocr_started_at = None
                    capture.ocr_heartbeat_at = None

        self._db.transaction(_recover)


def _ensure_aware(timestamp: dt.datetime | None) -> dt.datetime | None:
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp


def _build_ocr_payload(
    spans: list[tuple[str, float, list[int]]],
) -> tuple[str, list[dict]]:
    text_parts: list[str] = []
    ocr_spans: list[dict] = []
    offset = 0
    for idx, (text, conf, bbox) in enumerate(spans, start=1):
        text = text or ""
        start = offset
        end = start + len(text)
        text_parts.append(text)
        ocr_spans.append(
            {
                "span_id": f"S{idx}",
                "span_key": f"S{idx}",
                "start": start,
                "end": end,
                "conf": float(conf),
                "bbox": bbox,
                "text": text,
            }
        )
        offset = end + 1
    return "\n".join(text_parts), ocr_spans


def _extract_domain(window_title: str, ocr_text: str) -> str | None:
    pattern = re.compile(r"(?:https?://)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
    for source in (window_title or "", ocr_text or ""):
        match = pattern.search(source)
        if match:
            return match.group(1)
    return None
