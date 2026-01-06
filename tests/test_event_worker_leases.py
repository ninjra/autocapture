from __future__ import annotations

import datetime as dt

import numpy as np
from PIL import Image

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord
from autocapture.worker.event_worker import EventIngestWorker


class FakeOCR:
    def run(self, image: np.ndarray) -> list[tuple[str, float, list[int]]]:
        return [("hello", 0.9, [0, 0, 1, 0, 1, 1, 0, 1])]


def test_stale_processing_capture_is_reclaimed(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{db_path}"))
    db = DatabaseManager(config.database)

    image_path = tmp_path / "capture.png"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(image_path)

    capture_id = "capture-stale"
    stale_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=10)
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
                ocr_status="processing",
                ocr_started_at=stale_time,
                ocr_heartbeat_at=stale_time,
                ocr_attempts=0,
            )
        )

    worker = EventIngestWorker(config, db_manager=db, ocr_processor=FakeOCR())
    processed = worker.process_batch(limit=1)

    assert processed == 1
    with db.session() as session:
        capture = session.get(CaptureRecord, capture_id)
    assert capture is not None
    assert capture.ocr_status == "done"
