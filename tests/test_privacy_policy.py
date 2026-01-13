from __future__ import annotations

import numpy as np

from autocapture.capture.backends.monitor_utils import MonitorInfo
from autocapture.capture.orchestrator import CaptureOrchestrator
from autocapture.capture.raw_input import RawInputListener
from autocapture.config import AppConfig, DatabaseConfig, PrivacyConfig
from autocapture.storage.database import DatabaseManager
from autocapture.tracking.types import ForegroundContext


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


def _make_orchestrator(tmp_path, privacy: PrivacyConfig) -> CaptureOrchestrator:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:"),
        capture={"record_video": False, "data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        privacy=privacy,
    )
    db = DatabaseManager(config.database)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    raw_input = RawInputListener(idle_grace_ms=1000, on_activity=None, on_hotkey=None)
    backend = DummyBackend(frame)
    return CaptureOrchestrator(
        database=db,
        capture_config=config.capture,
        worker_config=config.worker,
        privacy_config=privacy,
        raw_input=raw_input,
        media_store=None,
        backend=backend,
    )


def test_denylisted_process_skips_capture(tmp_path, monkeypatch) -> None:
    privacy = PrivacyConfig(exclude_processes=["secrets.exe"])
    orchestrator = _make_orchestrator(tmp_path, privacy)

    monkeypatch.setattr(orchestrator, "_get_cursor_pos", lambda: (1, 1))
    monkeypatch.setattr(
        "autocapture.capture.orchestrator.get_foreground_context",
        lambda: ForegroundContext(process_name="Secrets.exe", window_title="Secrets", pid=1),
    )
    monkeypatch.setattr(
        "autocapture.capture.orchestrator.is_fullscreen_window",
        lambda hwnd: False,
    )
    monkeypatch.setattr(
        "autocapture.capture.orchestrator.get_screen_lock_status",
        lambda: (False, False),
    )

    captured = []

    def record_roi(item):
        captured.append(item)

    monkeypatch.setattr(orchestrator, "_enqueue_roi", record_roi)

    orchestrator._capture_tick()

    assert not captured


def test_denylisted_title_skips_capture(tmp_path, monkeypatch) -> None:
    privacy = PrivacyConfig(exclude_window_title_regex=[r"(?i)incognito"])
    orchestrator = _make_orchestrator(tmp_path, privacy)

    monkeypatch.setattr(orchestrator, "_get_cursor_pos", lambda: (1, 1))
    monkeypatch.setattr(
        "autocapture.capture.orchestrator.get_foreground_context",
        lambda: ForegroundContext(process_name="Browser.exe", window_title="Incognito Tab", pid=1),
    )
    monkeypatch.setattr(
        "autocapture.capture.orchestrator.is_fullscreen_window",
        lambda hwnd: False,
    )
    monkeypatch.setattr(
        "autocapture.capture.orchestrator.get_screen_lock_status",
        lambda: (False, False),
    )

    captured = []

    def record_roi(item):
        captured.append(item)

    monkeypatch.setattr(orchestrator, "_enqueue_roi", record_roi)

    orchestrator._capture_tick()

    assert not captured
