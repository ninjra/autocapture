from __future__ import annotations

import numpy as np

from autocapture.capture.backends.monitor_utils import MonitorInfo
from autocapture.capture.orchestrator import CaptureOrchestrator
from autocapture.capture.raw_input import RawInputListener
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager


class DummyBackend:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._monitors = [
            MonitorInfo(id="1", left=0, top=0, width=frame.shape[1], height=frame.shape[0])
        ]

    @property
    def monitors(self):
        return list(self._monitors)

    def grab_all(self):
        return {"1": self._frame}


def test_capture_loop_restarts_on_crash(tmp_path, monkeypatch) -> None:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:"),
        capture={"record_video": False, "data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
    )
    db = DatabaseManager(config.database)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
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

    orchestrator._running.set()
    calls = {"count": 0}

    def fail_once():
        calls["count"] += 1
        if calls["count"] >= 2:
            orchestrator._running.clear()
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator, "_run_capture_loop_once", fail_once)
    monkeypatch.setattr("autocapture.capture.orchestrator.time.sleep", lambda _: None)

    orchestrator._run_capture_loop()

    assert calls["count"] >= 2
