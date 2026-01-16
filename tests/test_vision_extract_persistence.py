from __future__ import annotations

import datetime as dt

from PIL import Image

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord
from autocapture.vision.types import ExtractionResult
from autocapture.worker.event_worker import EventIngestWorker


class FakeExtractor:
    def extract(self, _image):
        return ExtractionResult(
            text="screen transcript",
            spans=[],
            tags={
                "vision_extract": {
                    "schema_version": "v1",
                    "engine": "vlm",
                    "screen_summary": "summary",
                    "regions": [],
                    "visible_text": "screen transcript",
                    "parse_failed": False,
                    "tiles": [],
                }
            },
        )


def test_event_ingest_persists_vision_tags(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{db_path}"))
    db = DatabaseManager(config.database)

    image_path = tmp_path / "capture.png"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(image_path)

    capture_id = "capture-vision-1"
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

    worker = EventIngestWorker(config, db_manager=db, ocr_processor=FakeExtractor())
    assert worker.process_batch(limit=1) == 1

    with db.session() as session:
        event = session.get(EventRecord, capture_id)

    assert event is not None
    assert event.ocr_text == "screen transcript"
    assert event.tags.get("vision_extract", {}).get("engine") == "vlm"
