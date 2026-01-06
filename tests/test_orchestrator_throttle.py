from __future__ import annotations

import datetime as dt

from autocapture.capture.orchestrator import CaptureOrchestrator
from autocapture.capture.raw_input import RawInputListener
from autocapture.capture.backends.monitor_utils import MonitorInfo
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord


class DummyBackend:
    def __init__(self) -> None:
        self._monitors = [MonitorInfo(id="1", left=0, top=0, width=100, height=100)]

    @property
    def monitors(self):
        return list(self._monitors)

    def grab_all(self):
        return {}


def test_throttle_ignores_stale_processing(monkeypatch) -> None:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:"))
    db = DatabaseManager(config.database)
    stale_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=10)
    with db.session() as session:
        session.add(
            CaptureRecord(
                id="cap-1",
                captured_at=dt.datetime.now(dt.timezone.utc),
                image_path=None,
                foreground_process="test",
                foreground_window="test",
                monitor_id="1",
                is_fullscreen=False,
                ocr_status="processing",
                ocr_heartbeat_at=stale_time,
                ocr_attempts=0,
            )
        )

    raw_input = RawInputListener(idle_grace_ms=1000, on_activity=None, on_hotkey=None)
    orchestrator = CaptureOrchestrator(
        database=db,
        capture_config=config.capture,
        worker_config=config.worker,
        raw_input=raw_input,
        media_store=None,
        backend=DummyBackend(),
    )
    orchestrator._ocr_backlog_soft_limit = 1
    assert orchestrator._should_throttle_ocr() is False
