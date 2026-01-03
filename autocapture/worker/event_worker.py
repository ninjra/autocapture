"""Event ingest worker for OCR and event creation."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from sqlalchemy import select, update

from ..config import AppConfig
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import CaptureRecord, EventRecord


class OCRProcessor:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        self._warmup()

    def _warmup(self) -> None:
        sample = np.zeros((16, 16, 3), dtype=np.uint8)
        self._engine(sample)

    def run(self, image: np.ndarray) -> list[tuple[str, float, list[int]]]:
        results, _ = self._engine(image)
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
        if ocr_processor is None:
            self._ocr = OCRProcessor()
        else:
            self._ocr = ocr_processor

    def process_batch(self, limit: int = 25) -> int:
        processed = 0
        with self._db.session() as session:
            capture_ids = (
                session.execute(
                    select(CaptureRecord.id)
                    .where(CaptureRecord.ocr_status == "pending")
                    .order_by(CaptureRecord.captured_at.asc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )

        for capture_id in capture_ids:
            claimed = self._claim_capture(capture_id)
            if not claimed:
                continue
            try:
                self._ingest_capture(capture_id)
            except Exception as exc:
                self._log.exception("Failed to ingest capture %s: %s", capture_id, exc)
                self._mark_failed(capture_id)
                continue
            processed += 1
        return processed

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        poll_interval = self._config.worker.poll_interval_s
        while True:
            if stop_event and stop_event.is_set():
                return
            processed = self.process_batch()
            if processed == 0:
                time.sleep(poll_interval)

    def _claim_capture(self, capture_id: str) -> bool:
        with self._db.session() as session:
            result = session.execute(
                update(CaptureRecord)
                .where(
                    CaptureRecord.id == capture_id,
                    CaptureRecord.ocr_status == "pending",
                )
                .values(ocr_status="processing")
            )
            return result.rowcount == 1

    def _mark_failed(self, capture_id: str) -> None:
        with self._db.session() as session:
            capture = session.get(CaptureRecord, capture_id)
            if capture:
                capture.ocr_status = "failed"

    def _ingest_capture(self, capture_id: str) -> None:
        with self._db.session() as session:
            capture = session.get(CaptureRecord, capture_id)
            if not capture:
                return
            image = self._load_image(Path(capture.image_path))
            spans = self._ocr.run(image)
            ocr_text, ocr_spans = _build_ocr_payload(spans)
            screenshot_hash = _hash_file(Path(capture.image_path))
            event = EventRecord(
                event_id=capture.id,
                ts_start=capture.captured_at,
                ts_end=None,
                app_name=capture.foreground_process,
                window_title=capture.foreground_window,
                url=None,
                domain=_extract_domain(capture.foreground_window, ocr_text),
                screenshot_path=capture.image_path,
                screenshot_hash=screenshot_hash,
                ocr_text=ocr_text,
                ocr_spans=ocr_spans,
                embedding_vector=None,
                tags={},
            )
            session.add(event)
            capture.ocr_status = "done"

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        with path.open("rb") as handle:
            image = Image.open(handle)
            image = image.convert("RGB")
            return np.array(image)


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


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
