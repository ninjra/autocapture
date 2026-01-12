from __future__ import annotations

import datetime as dt
from pathlib import Path

from autocapture.capture.orchestrator import CaptureOrchestrator
from autocapture.capture.backends.monitor_utils import MonitorInfo
from autocapture.config import CaptureConfig, DatabaseConfig, WorkerConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord


class DummyBackend:
    def __init__(self) -> None:
        self._monitors = [MonitorInfo(id="m1", left=0, top=0, width=100, height=100)]

    @property
    def monitors(self):
        return self._monitors


class DummyRawInput:
    active_until_ts = 0

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def set_hotkey_callback(self, _callback) -> None:
        return None


def test_ocr_backlog_counts_all_pending(tmp_path: Path) -> None:
    db = DatabaseManager(DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    now = dt.datetime.now(dt.timezone.utc)

    def _seed(session) -> None:
        for idx, status in enumerate(["pending", "processing", "processing"], start=1):
            session.add(
                CaptureRecord(
                    id=f"cap-{idx}",
                    captured_at=now - dt.timedelta(minutes=idx),
                    image_path=str(tmp_path / f"cap-{idx}.webp"),
                    foreground_process="app",
                    foreground_window="win",
                    monitor_id="m1",
                    is_fullscreen=False,
                    ocr_status=status,
                    ocr_attempts=0,
                    ocr_heartbeat_at=now - dt.timedelta(days=1),
                )
            )

    db.transaction(_seed)

    orchestrator = CaptureOrchestrator(
        database=db,
        capture_config=CaptureConfig(staging_dir=tmp_path / "staging", data_dir=tmp_path),
        worker_config=WorkerConfig(),
        raw_input=DummyRawInput(),
        backend=DummyBackend(),
        ocr_backlog_check_s=0.0,
        media_store=None,
    )
    orchestrator._ocr_backlog_soft_limit = 2

    assert orchestrator._should_throttle_ocr() is True
    assert orchestrator._ocr_backlog_last_count == 3
