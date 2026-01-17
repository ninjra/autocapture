from __future__ import annotations

import time

from autocapture.config import RuntimeConfig
from autocapture.gpu_lease import GpuLease
from autocapture.runtime_governor import (
    FullscreenState,
    RuntimeGovernor,
    RuntimeMode,
    WindowMonitor,
)


class FakeRawInput:
    def __init__(self) -> None:
        self.last_input_ts = int(time.monotonic() * 1000)


class FakeWindowMonitor(WindowMonitor):
    def __init__(self) -> None:
        super().__init__()
        self.state = FullscreenState(False, None, None, None)

    def sample(self) -> FullscreenState:
        return self.state


def test_fullscreen_pause_triggers_gpu_release() -> None:
    config = RuntimeConfig()
    config.auto_pause.poll_hz = 50
    raw = FakeRawInput()
    monitor = FakeWindowMonitor()
    lease = GpuLease()
    releases: list[str] = []
    lease.register_release_hook("test", lambda reason: releases.append(reason))
    governor = RuntimeGovernor(
        config,
        raw_input=raw,
        window_monitor=monitor,
        gpu_lease=lease,
    )
    governor.start()
    monitor.state = FullscreenState(True, 1, "app.exe", "Full Screen")
    time.sleep(0.1)
    assert governor.current_mode == RuntimeMode.FULLSCREEN_HARD_PAUSE
    assert "fullscreen" in releases
    governor.stop()


def test_idle_transition() -> None:
    config = RuntimeConfig()
    config.auto_pause.poll_hz = 50
    raw = FakeRawInput()
    monitor = FakeWindowMonitor()
    lease = GpuLease()
    governor = RuntimeGovernor(
        config,
        raw_input=raw,
        window_monitor=monitor,
        gpu_lease=lease,
    )
    governor.start()
    raw.last_input_ts = int(time.monotonic() * 1000) - (config.qos.idle_grace_ms + 100)
    time.sleep(0.1)
    assert governor.current_mode == RuntimeMode.IDLE_DRAIN
    governor.stop()
