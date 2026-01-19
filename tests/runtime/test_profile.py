from __future__ import annotations

from autocapture.config import AppConfig
from autocapture.runtime_env import ProfileName, ProfileTuning, RuntimeEnvConfig, GpuMode
from autocapture.runtime_profile import ProfileScheduler


def test_profile_tuning_differs(tmp_path) -> None:
    foreground = ProfileTuning(
        max_workers=1,
        batch_size=4,
        poll_interval_ms=100,
        max_queue_depth=100,
        max_cpu_pct_hint=70,
    )
    idle = ProfileTuning(
        max_workers=2,
        batch_size=8,
        poll_interval_ms=500,
        max_queue_depth=200,
        max_cpu_pct_hint=30,
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

    assert scheduler.tuning(ProfileName.FOREGROUND) != scheduler.tuning(ProfileName.IDLE)
    assert scheduler.max_workers(ProfileName.FOREGROUND) == 1
    assert scheduler.max_workers(ProfileName.IDLE) == 2
