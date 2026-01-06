from __future__ import annotations

import numpy as np

from autocapture.capture.backends.monitor_utils import MonitorInfo
from autocapture.capture.orchestrator import CaptureOrchestrator
from autocapture.capture.raw_input import RawInputListener
from autocapture.config import AppConfig, CaptureConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.tracking.types import ForegroundContext


class DummyBackend:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._monitors = [
            MonitorInfo(
                id="1", left=0, top=0, width=frame.shape[1], height=frame.shape[0]
            )
        ]

    @property
    def monitors(self):
        return list(self._monitors)

    def grab_all(self):
        return {"1": self._frame}


def test_block_fullscreen_skips_capture(monkeypatch) -> None:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:"),
        capture=CaptureConfig(record_video=False),
    )
    db = DatabaseManager(config.database)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    raw_input = RawInputListener(idle_grace_ms=1000, on_activity=None, on_hotkey=None)
    backend = DummyBackend(frame)
    orchestrator = CaptureOrchestrator(
        database=db,
        capture_config=config.capture,
        worker_config=config.worker,
        raw_input=raw_input,
        media_store=None,
        backend=backend,
    )

    monkeypatch.setattr(orchestrator, "_get_cursor_pos", lambda: (1, 1))
    monkeypatch.setattr(
        "autocapture.capture.orchestrator.get_foreground_context",
        lambda: ForegroundContext(
            process_name="test", window_title="test", pid=1, hwnd=123
        ),
    )
    monkeypatch.setattr(
        "autocapture.capture.orchestrator.is_fullscreen_window",
        lambda hwnd: True,
    )

    captured = []

    def record_roi(item):
        captured.append(item)

    monkeypatch.setattr(orchestrator, "_enqueue_roi", record_roi)

    orchestrator._capture_tick()

    assert not captured
