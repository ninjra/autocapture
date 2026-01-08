from __future__ import annotations

import datetime as dt
from pathlib import Path

from PIL import Image

from autocapture.config import AppConfig, CaptureConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord, OCRSpanRecord
from autocapture.worker.event_worker import EventIngestWorker


class DummyOCR:
    def run(self, image):
        return [("hello", 0.9, [0, 0, 1, 1, 2, 2, 3, 3])]


def test_event_ingest_idempotent(tmp_path: Path) -> None:
    capture_dir = tmp_path / "data"
    capture_dir.mkdir(parents=True, exist_ok=True)
    image_path = tmp_path / "sample.webp"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)

    config = AppConfig(
        capture=CaptureConfig(staging_dir=tmp_path / "staging", data_dir=capture_dir),
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
    )
    db = DatabaseManager(config.database)
    now = dt.datetime.now(dt.timezone.utc)

    def _seed(session) -> None:
        capture = CaptureRecord(
            id="capture-1",
            captured_at=now,
            image_path=str(image_path),
            foreground_process="app",
            foreground_window="window",
            monitor_id="m1",
            is_fullscreen=False,
            ocr_status="pending",
        )
        session.add(capture)
        session.add(
            EventRecord(
                event_id=capture.id,
                ts_start=now,
                ts_end=None,
                app_name="app",
                window_title="window",
                url=None,
                domain=None,
                screenshot_path=str(image_path),
                screenshot_hash="hash",
                ocr_text="hello",
                embedding_vector=None,
                embedding_status="pending",
                embedding_model=config.embed.text_model,
                tags={},
            )
        )
        session.add(
            OCRSpanRecord(
                capture_id=capture.id,
                span_key="S1",
                start=0,
                end=5,
                text="hello",
                confidence=0.9,
                bbox=[0, 0, 1, 1],
            )
        )

    db.transaction(_seed)

    worker = EventIngestWorker(config, db_manager=db, ocr_processor=DummyOCR())
    worker.process_batch(limit=1)

    with db.session() as session:
        capture = session.get(CaptureRecord, "capture-1")
        event = session.get(EventRecord, "capture-1")

    assert capture is not None
    assert event is not None
    assert capture.ocr_status == "done"
    assert capture.ocr_last_error is None
