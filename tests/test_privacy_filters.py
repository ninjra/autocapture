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


def _make_orchestrator(
    tmp_path, privacy: PrivacyConfig, *, roi_w: int, roi_h: int
) -> CaptureOrchestrator:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:"),
        capture={
            "record_video": False,
            "data_dir": tmp_path,
            "staging_dir": tmp_path / "staging",
        },
        privacy=privacy,
    )
    db = DatabaseManager(config.database)
    frame = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
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
        roi_w=roi_w,
        roi_h=roi_h,
    )


def _patch_foreground(monkeypatch, *, process_name: str, window_title: str) -> None:
    monkeypatch.setattr(
        "autocapture.capture.orchestrator.get_foreground_context",
        lambda: ForegroundContext(
            process_name=process_name, window_title=window_title, pid=1, hwnd=None
        ),
    )


def _patch_environment(monkeypatch, orchestrator: CaptureOrchestrator) -> None:
    monkeypatch.setattr(orchestrator, "_get_cursor_pos", lambda: (1, 1))
    monkeypatch.setattr(
        "autocapture.capture.orchestrator.is_fullscreen_window",
        lambda hwnd: False,
    )
    monkeypatch.setattr(
        "autocapture.capture.orchestrator.get_screen_lock_status",
        lambda: (False, False),
    )


def test_exclude_process_skips_capture(tmp_path, monkeypatch) -> None:
    privacy = PrivacyConfig(exclude_processes=["secret.exe"])
    orchestrator = _make_orchestrator(tmp_path, privacy, roi_w=32, roi_h=32)
    _patch_environment(monkeypatch, orchestrator)
    _patch_foreground(monkeypatch, process_name="secret.exe", window_title="Top Secret")

    captured = []
    video_calls = []

    monkeypatch.setattr(orchestrator, "_enqueue_roi", lambda item: captured.append(item))
    monkeypatch.setattr(
        orchestrator, "_enqueue_video", lambda *args, **kwargs: video_calls.append(1)
    )

    orchestrator._capture_tick()

    assert not captured
    assert not video_calls


def test_exclude_window_title_regex_skips_capture(tmp_path, monkeypatch) -> None:
    privacy = PrivacyConfig(exclude_window_title_regex=[r".*Private.*"])
    orchestrator = _make_orchestrator(tmp_path, privacy, roi_w=32, roi_h=32)
    _patch_environment(monkeypatch, orchestrator)
    _patch_foreground(monkeypatch, process_name="browser.exe", window_title="Private Browsing")

    captured = []
    video_calls = []

    monkeypatch.setattr(orchestrator, "_enqueue_roi", lambda item: captured.append(item))
    monkeypatch.setattr(
        orchestrator, "_enqueue_video", lambda *args, **kwargs: video_calls.append(1)
    )

    orchestrator._capture_tick()

    assert not captured
    assert not video_calls


def test_exclude_monitors_skips_capture(tmp_path, monkeypatch) -> None:
    privacy = PrivacyConfig(exclude_monitors=["1"])
    orchestrator = _make_orchestrator(tmp_path, privacy, roi_w=32, roi_h=32)
    _patch_environment(monkeypatch, orchestrator)
    _patch_foreground(monkeypatch, process_name="app.exe", window_title="Work")

    captured = []
    video_calls = []

    monkeypatch.setattr(orchestrator, "_enqueue_roi", lambda item: captured.append(item))
    monkeypatch.setattr(
        orchestrator, "_enqueue_video", lambda *args, **kwargs: video_calls.append(1)
    )

    orchestrator._capture_tick()

    assert not captured
    assert not video_calls


def test_exclude_regions_masks_pixels(tmp_path, monkeypatch) -> None:
    privacy = PrivacyConfig(
        exclude_regions=[{"monitor_id": "1", "x": 0, "y": 0, "width": 2, "height": 2}]
    )
    orchestrator = _make_orchestrator(tmp_path, privacy, roi_w=8, roi_h=8)
    _patch_environment(monkeypatch, orchestrator)
    _patch_foreground(monkeypatch, process_name="app.exe", window_title="Work")

    frame = np.full((8, 8, 3), 255, dtype=np.uint8)
    orchestrator._backend._frame = frame

    captured = []

    monkeypatch.setattr(orchestrator, "_enqueue_roi", lambda item: captured.append(item))
    monkeypatch.setattr(orchestrator, "_enqueue_video", lambda *args, **kwargs: None)

    orchestrator._capture_tick()

    assert captured
    roi = captured[0].image
    assert (roi[:2, :2] == 0).all()
    assert (roi[2:, 2:] == 255).all()
