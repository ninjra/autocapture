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
    monitor.state = FullscreenState(True, 1, "app.exe", "Full Screen")
    governor.tick()
    assert governor.current_mode == RuntimeMode.FULLSCREEN_HARD_PAUSE
    assert "fullscreen" in releases


def test_idle_transition() -> None:
    config = RuntimeConfig()
    raw = FakeRawInput()
    monitor = FakeWindowMonitor()
    lease = GpuLease()
    governor = RuntimeGovernor(
        config,
        raw_input=raw,
        window_monitor=monitor,
        gpu_lease=lease,
    )
    raw.last_input_ts = int(time.monotonic() * 1000) - (config.qos.idle_grace_ms + 100)
    governor.tick()
    assert governor.current_mode == RuntimeMode.IDLE_DRAIN


def test_snapshot_reason_and_since_ts() -> None:
    config = RuntimeConfig()
    raw = FakeRawInput()
    monitor = FakeWindowMonitor()
    governor = RuntimeGovernor(config, raw_input=raw, window_monitor=monitor)
    snap_initial = governor.snapshot()
    monitor.state = FullscreenState(True, 1, "app.exe", "Full Screen")
    governor.tick()
    snap_full = governor.snapshot()
    assert snap_full.mode == RuntimeMode.FULLSCREEN_HARD_PAUSE
    assert snap_full.reason == "fullscreen"
    assert snap_full.since_ts >= snap_initial.since_ts
