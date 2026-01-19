"""Runtime context builder for pause/device/profile switches."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig
from .runtime_device import DeviceManager
from .runtime_env import RuntimeEnvConfig, load_runtime_env
from .runtime_pause import PauseController
from .runtime_profile import ExecutionProfile, ProfileScheduler


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    env: RuntimeEnvConfig
    profile: ExecutionProfile
    scheduler: ProfileScheduler
    pause: PauseController
    device: DeviceManager


def build_runtime_context(
    config: AppConfig,
    runtime_env: RuntimeEnvConfig | None = None,
) -> RuntimeContext:
    env = runtime_env or load_runtime_env()
    scheduler = ProfileScheduler(config, runtime_env=env)
    profile = scheduler.profile(env.profile)
    pause = PauseController(
        env.pause_latch_path,
        env.pause_reason_path,
        poll_interval_s=profile.poll_interval_s,
        redact_window_titles=env.redact_window_titles,
    )
    device = DeviceManager(env, pause)
    return RuntimeContext(env=env, profile=profile, scheduler=scheduler, pause=pause, device=device)
