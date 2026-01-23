from __future__ import annotations

import time

from autocapture.config import RuntimeConfig, AppConfig
from autocapture.gpu_lease import GpuLease
from autocapture.runtime_env import ProfileName, ProfileTuning, RuntimeEnvConfig, GpuMode
from autocapture.runtime_governor import (
    FullscreenState,
    RuntimeGovernor,
    RuntimeMode,
    WindowMonitor,
)
from autocapture.runtime_profile import ProfileScheduler


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


def test_max_cpu_pct_hint_from_scheduler(tmp_path) -> None:
    foreground = ProfileTuning(
        max_workers=1,
        batch_size=4,
        poll_interval_ms=100,
        max_queue_depth=100,
        max_cpu_pct_hint=33,
    )
    idle = ProfileTuning(
        max_workers=1,
        batch_size=4,
        poll_interval_ms=100,
        max_queue_depth=100,
        max_cpu_pct_hint=12,
    )
    env = RuntimeEnvConfig(
        gpu_mode=GpuMode.AUTO,
        profile=ProfileName.FOREGROUND,
        profile_override=True,
        runtime_dir=tmp_path,
        bench_output_dir=tmp_path / "bench",
        redact_window_titles=True,
        cuda_device_index=0,
        cuda_visible_devices="0",
        foreground_tuning=foreground,
        idle_tuning=idle,
    )
    scheduler = ProfileScheduler(AppConfig(), runtime_env=env)
    governor = RuntimeGovernor(RuntimeConfig(), profile_scheduler=scheduler)
    assert governor.max_cpu_pct_hint() == 33
    governor.set_profile_override(ProfileName.IDLE)
    assert governor.max_cpu_pct_hint() == 12
