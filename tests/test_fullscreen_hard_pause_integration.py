from __future__ import annotations

import datetime as dt
import threading
import time

from PIL import Image

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.runtime_governor import FullscreenState, RuntimeGovernor, WindowMonitor
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord
from autocapture.worker.embedding_worker import EmbeddingWorker
from autocapture.worker.event_worker import EventIngestWorker


class FakeRawInput:
    def __init__(self) -> None:
        self.last_input_ts = int(time.monotonic() * 1000)


class FakeWindowMonitor(WindowMonitor):
    def __init__(self) -> None:
        super().__init__()
        self.state = FullscreenState(True, 1, "game.exe", "Fullscreen")

    def sample(self) -> FullscreenState:
        return self.state


class FakeOCR:
    def run(self, _image):
        return [("hello", 0.9, [0, 0, 1, 0, 1, 1, 0, 1])]


def _run_worker(worker, stop_event: threading.Event) -> None:
    worker.run_forever(stop_event=stop_event)


def test_fullscreen_hard_pause_blocks_worker_writes(tmp_path) -> None:
    db_path = tmp_path / "pause.db"
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{db_path}"))
    config.capture.data_dir = tmp_path
    config.embed.text_model = "local-test"
    config.runtime.auto_pause.enabled = True
    config.runtime.auto_pause.fullscreen_hard_pause_enabled = True
    config.runtime.qos.profile_active.sleep_ms = 10

    db = DatabaseManager(config.database)

    image_path = tmp_path / "capture.png"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(image_path)

    capture_id = "capture-fullscreen"
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            CaptureRecord(
                id=capture_id,
                captured_at=now,
                image_path=str(image_path),
                foreground_process="test.exe",
                foreground_window="Window",
                monitor_id="monitor-1",
                is_fullscreen=False,
                ocr_status="pending",
                ocr_attempts=0,
            )
        )
        session.add(
            EventRecord(
                event_id="event-1",
                ts_start=now,
                ts_end=None,
                app_name="Docs",
                window_title="Notes",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="text",
                embedding_status="pending",
                embedding_attempts=0,
                tags={},
            )
        )

    raw = FakeRawInput()
    monitor = FakeWindowMonitor()
    governor = RuntimeGovernor(config.runtime, raw_input=raw, window_monitor=monitor)
    governor.tick()

    ocr_worker = EventIngestWorker(
        config,
        db_manager=db,
        ocr_processor=FakeOCR(),
        runtime_governor=governor,
    )
    embed_worker = EmbeddingWorker(config, db_manager=db, runtime_governor=governor)

    stop_event = threading.Event()
    threads = [
        threading.Thread(target=_run_worker, args=(ocr_worker, stop_event), daemon=True),
        threading.Thread(target=_run_worker, args=(embed_worker, stop_event), daemon=True),
    ]
    for thread in threads:
        thread.start()

    time.sleep(0.05)
    stop_event.set()
    for thread in threads:
        thread.join(timeout=1.0)

    with db.session() as session:
        capture = session.get(CaptureRecord, capture_id)
        event = session.get(EventRecord, "event-1")

    assert capture is not None
    assert capture.ocr_status == "pending"
    assert capture.ocr_attempts == 0
    assert event is not None
    assert event.embedding_status == "pending"
    assert event.embedding_attempts == 0
