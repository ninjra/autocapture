from __future__ import annotations

import datetime as dt

import numpy as np
from PIL import Image

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord
from autocapture.worker.event_worker import EventIngestWorker


class FakeOCR:
    def run(self, image: np.ndarray) -> list[tuple[str, float, list[int]]]:
        return [("hello world", 0.9, [0, 0, 10, 0, 10, 10, 0, 10])]


def test_event_ingest_worker_creates_event(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{db_path}"))
    db = DatabaseManager(config.database)

    image_path = tmp_path / "capture.png"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(image_path)

    capture_id = "capture-1"
    with db.session() as session:
        session.add(
            CaptureRecord(
                id=capture_id,
                captured_at=dt.datetime.now(dt.timezone.utc),
                image_path=str(image_path),
                foreground_process="test.exe",
                foreground_window="Example Window",
                monitor_id="monitor-1",
                is_fullscreen=False,
                ocr_status="pending",
            )
        )

    worker = EventIngestWorker(config, db_manager=db, ocr_processor=FakeOCR())
    processed = worker.process_batch(limit=1)
    assert processed == 1

    with db.session() as session:
        capture = session.get(CaptureRecord, capture_id)
        event = session.get(EventRecord, capture_id)

    assert capture is not None
    assert capture.ocr_status == "done"
    assert event is not None
    assert event.ocr_text
    assert event.ocr_spans
    span = event.ocr_spans[0]
    assert span["span_id"]
    assert span["start"] <= span["end"]
