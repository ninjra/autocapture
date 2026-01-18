"""Event ingest worker for OCR and event creation."""

from __future__ import annotations

import datetime as dt
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

from ..agents import AGENT_JOB_ENRICH_EVENT
from ..agents.jobs import AgentJobQueue
from ..config import AppConfig, CaptureConfig, OCRConfig
from ..runtime_governor import RuntimeGovernor
from ..runtime_pause import PauseController, paused_guard
from ..image_utils import ensure_rgb, hash_rgb_image
from ..indexing.lexical_index import LexicalIndex
from ..logging_utils import get_logger
from ..media.store import MediaStore
from ..observability.metrics import ocr_backlog, ocr_latency_ms, worker_errors_total
from ..observability.otel import otel_span, record_histogram
from ..storage.database import DatabaseManager
from ..storage.models import CaptureRecord, EmbeddingRecord, EventRecord, OCRSpanRecord
from ..vision.extractors import ScreenExtractorRouter
from ..vision.layout import build_layout
from ..vision.paddle_layout import PaddleLayoutExtractor
from ..vision.hdr import apply_hdr_tone_mapping
from ..vision.ui_grounding import UIGroundingRouter
from ..vision.types import ExtractionResult, VISION_SCHEMA_VERSION, build_ocr_payload
from ..enrichment.sql_artifacts import SqlArtifacts, extract_sql_artifacts
from ..text.normalize import normalize_text


@dataclass(frozen=True)
class CapturePayload:
    capture_id: str
    captured_at: dt.datetime
    image_path: str | None
    focus_path: str | None
    foreground_process: str
    foreground_window: str
    monitor_id: str
    is_fullscreen: bool
    monitor_bounds: list[int] | None
    frame_hash: str | None
    privacy_flags: dict


def _build_rapidocr_kwargs(use_cuda: bool) -> dict[str, object]:
    from ..vision.rapidocr import _build_rapidocr_kwargs as build_kwargs

    return build_kwargs(use_cuda)


def _select_onnx_provider(config: OCRConfig, providers: Iterable[str]) -> tuple[str | None, bool]:
    from ..vision.rapidocr import select_onnx_provider

    return select_onnx_provider(config, providers)


class LegacyOCRAdapter:
    def __init__(self, ocr_processor: object) -> None:
        self._processor = ocr_processor

    def extract(self, image: np.ndarray) -> ExtractionResult:
        spans = self._processor.run(image)
        text, ocr_spans = build_ocr_payload(spans)
        tags = {
            "vision_extract": {
                "schema_version": VISION_SCHEMA_VERSION,
                "engine": "legacy-ocr",
                "screen_summary": "",
                "regions": [],
                "visible_text": text,
                "content_flags": [],
                "tables_detected": None,
                "spreadsheets_detected": None,
                "parse_failed": False,
                "parse_format": "legacy",
                "tiles": [],
            }
        }
        return ExtractionResult(text=text, spans=ocr_spans, tags=tags)


class EventIngestWorker:
    def __init__(
        self,
        config: AppConfig,
        db_manager: DatabaseManager | None = None,
        ocr_processor: Optional[object] = None,
        runtime_governor: RuntimeGovernor | None = None,
        pause_controller: PauseController | None = None,
    ) -> None:
        self._config = config
        self._db = db_manager or DatabaseManager(config.database)
        self._log = get_logger("worker.event_ingest")
        self._runtime = runtime_governor
        self._pause = pause_controller
        self._lexical = LexicalIndex(self._db)
        self._media_store = MediaStore(config.capture, config.encryption)
        self._agent_jobs = AgentJobQueue(self._db)
        self._lease_timeout_s = config.worker.ocr_lease_ms / 1000
        self._max_attempts = config.worker.ocr_max_attempts
        self._max_task_runtime_s = config.worker.max_task_runtime_s
        if ocr_processor is None:
            self._extractor = ScreenExtractorRouter(config, runtime_governor=runtime_governor)
        else:
            if hasattr(ocr_processor, "extract"):
                self._extractor = ocr_processor
            elif hasattr(ocr_processor, "run"):
                self._extractor = LegacyOCRAdapter(ocr_processor)
            else:
                raise ValueError("ocr_processor must implement extract() or run()")
        self._ui_grounding = UIGroundingRouter(config, runtime_governor=runtime_governor)
        self._ppstructure = PaddleLayoutExtractor(config.ocr)

    def _allow_work(self) -> bool:
        if not self._runtime:
            return True
        if self._runtime.allow_workers():
            return True
        self._log.debug("OCR worker paused by runtime governor")
        return False

    def process_batch(self, limit: Optional[int] = None) -> int:
        if paused_guard(self._pause):
            return 0
        if not self._allow_work():
            return 0
        processed = 0
        if not self._allow_work():
            return 0
        self._recover_stale_captures()
        if limit is None:
            limit = self._config.ocr.batch_size
            if self._runtime:
                profile = self._runtime.qos_profile()
                if profile.ocr_batch_size:
                    limit = profile.ocr_batch_size
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
            if not self._allow_work():
                break
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
        if self._allow_work():
            self._recover_stale_captures()
        backoff_s = 1.0
        while True:
            if stop_event and stop_event.is_set():
                return
            if paused_guard(self._pause, stop_event):
                return
            if not self._allow_work():
                sleep_ms = self._runtime.qos_budget().sleep_ms if self._runtime else int(1000)
                time.sleep(max(0.01, sleep_ms / 1000.0))
                continue
            try:
                processed = self.process_batch()
                backoff_s = 1.0
            except Exception as exc:
                self._log.exception("OCR worker loop failed: {}", exc)
                worker_errors_total.labels("ocr").inc()
                if stop_event and stop_event.is_set():
                    return
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 30.0)
                continue
            if processed == 0:
                poll_interval = self._config.worker.poll_interval_s
                if self._runtime:
                    poll_interval = self._runtime.poll_interval_s(poll_interval)
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
        if not self._allow_work():
            return False

        existing_event = self._load_event(capture_id)
        existing_spans = self._load_spans(capture_id) if existing_event else []
        if existing_event and existing_event.ocr_text and existing_spans:
            normalized_text = (
                normalize_text(existing_event.ocr_text)
                if self._config.features.enable_normalized_indexing
                else None
            )
            self._persist_ocr_results(
                capture,
                existing_event.ocr_text or "",
                existing_spans,
                event_existing=True,
                screenshot_hash=capture.frame_hash,
                normalized_text=normalized_text,
            )
            self._lexical.upsert_event(existing_event)
            self._enqueue_enrichment(existing_event.event_id)
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
            args=(capture_id, stop_event, time.monotonic()),
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            image = self._load_image(path)
            image, hdr_tags = apply_hdr_tone_mapping(
                image, enabled=self._config.capture.hdr_enabled
            )
            frame_height, frame_width = image.shape[:2]
            capture_meta = _build_capture_meta(
                capture,
                frame_width,
                frame_height,
                hdr_tags,
                self._config.capture,
            )
            paused_guard(self._pause)
            ocr_start = time.monotonic()
            with otel_span("extract_ocr", {"stage_name": "extract_ocr"}):
                result = self._extractor.extract(image)
            record_histogram(
                "extract_ocr_ms",
                (time.monotonic() - ocr_start) * 1000,
                {"stage_name": "extract_ocr"},
            )
            ocr_text = result.text
            ocr_spans = result.spans
            engine = (
                (result.tags or {}).get("vision_extract", {}).get("engine")
                if isinstance(result.tags, dict)
                else None
            )
            tags = _merge_tags(result.tags or {}, {"capture_meta": capture_meta, "hdr": hdr_tags})
            layout_blocks: list[dict] = []
            layout_md = ""
            layout_source = "disabled"
            if self._config.ocr.layout_enabled:
                layout_blocks, layout_md = build_layout(ocr_spans)
                layout_source = "ocr_layout"
            paused_guard(self._pause)
            pp_layout = self._ppstructure.extract(image)
            if pp_layout and pp_layout.tags.get("status") == "ok" and pp_layout.blocks:
                layout_blocks = pp_layout.blocks
                layout_md = pp_layout.markdown
                layout_source = "ppstructure"
            tags = _merge_tags(
                tags,
                {
                    "layout_blocks": layout_blocks,
                    "layout_md": layout_md,
                    "layout_source": layout_source,
                    "ppstructure": pp_layout.tags if pp_layout else {},
                },
            )
            paused_guard(self._pause)
            ui_result = self._ui_grounding.extract(image)
            tags = _merge_tags(tags, {"ui_elements": ui_result.tags})
            sql_artifacts = extract_sql_artifacts(
                ocr_text,
                (result.tags or {}).get("vision_extract", {}).get("regions", []),
            )
            frame_hash = capture.frame_hash
            if self._config.features.enable_frame_hash and not frame_hash:
                frame_hash = hash_rgb_image(image)
            normalized_text = (
                normalize_text(ocr_text)
                if self._config.features.enable_normalized_indexing
                else None
            )
            self._persist_ocr_results(
                capture,
                ocr_text,
                ocr_spans,
                event_existing=bool(existing_event),
                screenshot_hash=frame_hash,
                tags=tags,
                sql_artifacts=sql_artifacts,
                engine=engine,
                frame_hash=frame_hash,
                normalized_text=normalized_text,
            )
            event = self._load_event(capture_id)
            if event:
                with otel_span("index_lexical", {"stage_name": "index_lexical"}):
                    self._lexical.upsert_event(event)
                self._enqueue_enrichment(event.event_id)
                if sql_artifacts.artifact_text:
                    self._lexical.upsert_agent_text(event.event_id, sql_artifacts.artifact_text)
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

    def _enqueue_enrichment(self, event_id: str) -> None:
        if not self._config.agents.enabled:
            return
        prompt_revision = "EVENT_ENRICHMENT:v1"
        model_id = _llm_model_id(self._config)
        schema_version = "v2"
        job_key = f"enrich:{event_id}:{schema_version}:{prompt_revision}:{model_id}"
        self._agent_jobs.enqueue(
            job_key=job_key,
            job_type=AGENT_JOB_ENRICH_EVENT,
            event_id=event_id,
            payload={"event_id": event_id},
            max_attempts=3,
            max_pending=self._config.agents.max_pending_jobs,
        )

    def _load_capture(self, capture_id: str) -> CapturePayload | None:
        with self._db.session() as session:
            capture = session.get(CaptureRecord, capture_id)
            if not capture:
                return None
            return CapturePayload(
                capture_id=capture.id,
                captured_at=capture.captured_at,
                image_path=capture.image_path,
                focus_path=capture.focus_path,
                foreground_process=capture.foreground_process,
                foreground_window=capture.foreground_window,
                monitor_id=capture.monitor_id,
                is_fullscreen=capture.is_fullscreen,
                monitor_bounds=capture.monitor_bounds,
                frame_hash=getattr(capture, "frame_hash", None),
                privacy_flags=(capture.privacy_flags or {}),
            )

    def _load_event(self, event_id: str) -> EventRecord | None:
        with self._db.session() as session:
            return session.get(EventRecord, event_id)

    def _load_spans(self, capture_id: str) -> list[dict]:
        with self._db.session() as session:
            rows = (
                session.execute(
                    select(OCRSpanRecord)
                    .where(OCRSpanRecord.capture_id == capture_id)
                    .order_by(OCRSpanRecord.start.asc())
                )
                .scalars()
                .all()
            )
        return [
            {
                "span_key": row.span_key,
                "span_id": row.span_key,
                "start": row.start,
                "end": row.end,
                "conf": row.confidence,
                "bbox": row.bbox,
                "text": row.text,
            }
            for row in rows
        ]

    def _heartbeat_loop(
        self,
        capture_id: str,
        stop_event: threading.Event,
        start_ts: float,
    ) -> None:
        interval = max(self._lease_timeout_s / 3, 1.0)
        warned = False
        while not stop_event.wait(interval):
            if not self._allow_work():
                return
            if time.monotonic() - start_ts >= self._max_task_runtime_s:
                if not warned:
                    self._log.warning(
                        "OCR heartbeat exceeded max runtime; stopping so lease can be reclaimed."
                    )
                    warned = True
                return
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
        sql_artifacts: SqlArtifacts | None = None,
        screenshot_hash: str | None = None,
        tags: dict | None = None,
        engine: str | None = None,
        frame_hash: str | None = None,
        normalized_text: str | None = None,
    ) -> None:
        capture_id = getattr(capture, "capture_id", None) or getattr(capture, "id")
        focus_reference = self._config.capture.focus_crop_reference
        focus_path = capture.focus_path if focus_reference == "event" else None
        extra_tags: dict = {}
        if capture.focus_path and focus_reference == "tags":
            extra_tags = {"capture": {"focus_path": capture.focus_path}}
        derived_tags = tags or {}
        if sql_artifacts and (sql_artifacts.code_blocks or sql_artifacts.sql_statements):
            derived_tags = _merge_tags(derived_tags, {"sql_artifacts": sql_artifacts.as_tags()})
        if capture.privacy_flags:
            derived_tags = _merge_tags(derived_tags, {"privacy_flags": capture.privacy_flags})

        def _write(session) -> None:
            if not event_existing:
                event = EventRecord(
                    event_id=capture_id,
                    ts_start=capture.captured_at,
                    ts_end=None,
                    app_name=capture.foreground_process,
                    window_title=capture.foreground_window,
                    url=None,
                    domain=_extract_domain(capture.foreground_window, ocr_text),
                    screenshot_path=capture.image_path,
                    focus_path=focus_path,
                    screenshot_hash=screenshot_hash or "",
                    frame_hash=frame_hash,
                    ocr_text=ocr_text,
                    ocr_text_normalized=normalized_text,
                    embedding_vector=None,
                    embedding_status="pending",
                    embedding_model=self._config.embed.text_model,
                    tags=_merge_tags(extra_tags, derived_tags),
                )
                session.add(event)
                if session.bind and session.bind.dialect.name == "sqlite":
                    session.flush()
            else:
                event = session.get(EventRecord, capture_id)
                if event and not event.ocr_text:
                    event.ocr_text = ocr_text
                    if screenshot_hash and not event.screenshot_hash:
                        event.screenshot_hash = screenshot_hash
                    if frame_hash and not getattr(event, "frame_hash", None):
                        event.frame_hash = frame_hash
                    if focus_path and not event.focus_path:
                        event.focus_path = focus_path
                if event and normalized_text and not event.ocr_text_normalized:
                    event.ocr_text_normalized = normalized_text
                if event and derived_tags:
                    event.tags = _merge_tags(
                        _merge_tags(event.tags or {}, extra_tags), derived_tags
                    )
                elif event and extra_tags:
                    event.tags = _merge_tags(event.tags or {}, extra_tags)
            if ocr_spans:
                self._upsert_spans(
                    session,
                    capture_id,
                    ocr_spans,
                    engine=engine,
                    frame_hash=frame_hash,
                    monitor_bounds=capture.monitor_bounds,
                )
                span_map = {
                    span.span_key: span
                    for span in session.execute(
                        select(OCRSpanRecord).where(
                            OCRSpanRecord.capture_id == capture_id,
                            OCRSpanRecord.span_key.in_(
                                [str(span.get("span_key")) for span in ocr_spans]
                            ),
                        )
                    )
                    .scalars()
                    .all()
                }
                self._upsert_embeddings(session, capture_id, span_map, frame_hash)
            record = session.get(CaptureRecord, capture_id)
            if record:
                record.ocr_status = "done"
                record.ocr_last_error = None
                if frame_hash and not record.frame_hash:
                    record.frame_hash = frame_hash

        self._db.transaction(_write)

    def _upsert_spans(
        self,
        session,
        capture_id: str,
        ocr_spans: list[dict],
        *,
        engine: str | None,
        frame_hash: str | None,
        monitor_bounds: list[int] | None,
    ) -> None:
        rows = [
            {
                "capture_id": capture_id,
                "span_key": str(span.get("span_key")),
                "start": int(span.get("start", 0)),
                "end": int(span.get("end", 0)),
                "text": str(span.get("text", "")),
                "confidence": float(span.get("conf", 0.0)),
                "bbox": _clamp_bbox(span.get("bbox", []), monitor_bounds),
                "engine": engine,
                "frame_hash": frame_hash,
                "schema_version": "v1",
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
        self,
        session,
        capture_id: str,
        span_map: dict[str, OCRSpanRecord],
        frame_hash: str | None,
    ) -> None:
        rows = []
        for span_key, span in span_map.items():
            rows.append(
                {
                    "capture_id": capture_id,
                    "vector": None,
                    "model": self._config.embed.text_model,
                    "status": "pending",
                    "span_key": span_key,
                    "frame_hash": frame_hash,
                }
            )
        if not rows:
            return
        dialect = session.bind.dialect.name if session.bind else ""
        if dialect == "sqlite":
            stmt = (
                sqlite_insert(EmbeddingRecord)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["capture_id", "span_key", "model"])
            )
            session.execute(stmt)
        elif dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = (
                pg_insert(EmbeddingRecord)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["capture_id", "span_key", "model"])
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
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=self._lease_timeout_s)

        def _recover(session) -> None:
            rows = (
                session.execute(
                    select(CaptureRecord).where(CaptureRecord.ocr_status == "processing")
                )
                .scalars()
                .all()
            )
            for capture in rows:
                heartbeat = (
                    capture.ocr_heartbeat_at or capture.ocr_started_at or capture.captured_at
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


def _merge_tags(existing: dict, incoming: dict) -> dict:
    merged = dict(existing or {})
    for key, value in (incoming or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_tags(merged[key], value)
        else:
            merged[key] = value
    return merged


def _extract_domain(window_title: str, ocr_text: str) -> str | None:
    pattern = re.compile(r"(?:https?://)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
    for source in (window_title or "", ocr_text or ""):
        match = pattern.search(source)
        if match:
            return match.group(1)
    return None


def _build_capture_meta(
    capture: CapturePayload,
    frame_width: int,
    frame_height: int,
    hdr_tags: dict,
    capture_config: CaptureConfig,
) -> dict:
    meta: dict[str, object] = {
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "color_space": (
            hdr_tags.get("color_space", "sRGB") if isinstance(hdr_tags, dict) else "sRGB"
        ),
        "hdr": hdr_tags,
    }
    if capture_config.multi_monitor_enabled:
        meta["monitor_id"] = capture.monitor_id
        meta["monitor_bounds"] = capture.monitor_bounds
        if capture.monitor_bounds and len(capture.monitor_bounds) >= 4:
            try:
                bounds_width = int(capture.monitor_bounds[2])
                bounds_height = int(capture.monitor_bounds[3])
            except (TypeError, ValueError):
                bounds_width = 0
                bounds_height = 0
            if bounds_width > 0 and bounds_height > 0:
                scale_x = round(frame_width / bounds_width, 6)
                scale_y = round(frame_height / bounds_height, 6)
                meta["dpi_scale"] = [scale_x, scale_y]
    return meta


def _clamp_bbox(bbox: object, monitor_bounds: list[int] | None) -> object:
    if not monitor_bounds or not bbox:
        return bbox
    if len(monitor_bounds) < 4:
        return bbox
    width = int(monitor_bounds[2])
    height = int(monitor_bounds[3])
    if width <= 0 or height <= 0:
        return bbox
    if isinstance(bbox, dict):
        clamped = dict(bbox)
        for key, limit in (("x0", width), ("x1", width), ("y0", height), ("y1", height)):
            if key in clamped:
                try:
                    value = float(clamped[key])
                    clamped[key] = max(0, min(int(round(value)), limit))
                except (TypeError, ValueError):
                    continue
        return clamped
    if isinstance(bbox, list) and bbox:
        clamped_list: list[int] = []
        for idx, value in enumerate(bbox):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                clamped_list.append(0)
                continue
            limit = width if idx % 2 == 0 else height
            clamped_list.append(max(0, min(int(round(numeric)), limit)))
        return clamped_list
    return bbox


def _llm_model_id(config: AppConfig) -> str:
    provider = config.llm.provider
    if provider == "openai":
        model = config.llm.openai_model
    elif provider == "openai_compatible":
        model = config.llm.openai_compatible_model
    else:
        model = config.llm.ollama_model
    return f"{provider}:{model}"
