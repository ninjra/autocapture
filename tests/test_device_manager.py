from __future__ import annotations

import pytest

from autocapture.runtime_device import DeviceKind, DeviceManager
from autocapture.runtime_env import GpuMode, ProfileName, RuntimeEnvConfig


class FakePause:
    def __init__(self, paused: bool) -> None:
        self._paused = paused

    def is_paused(self) -> bool:
        return self._paused


def _env(tmp_path, gpu_mode: GpuMode) -> RuntimeEnvConfig:
    return RuntimeEnvConfig(
        gpu_mode=gpu_mode,
        profile=ProfileName.FOREGROUND,
        profile_override=False,
        runtime_dir=tmp_path,
        bench_output_dir=tmp_path / "bench",
        redact_window_titles=True,
        cuda_device_index=0,
        cuda_visible_devices="0",
    )


@pytest.mark.parametrize(
    "gpu_mode,paused,cuda_available,expect_kind,expect_error",
    [
        (GpuMode.OFF, False, False, DeviceKind.CPU, False),
        (GpuMode.OFF, True, True, DeviceKind.CPU, False),
        (GpuMode.AUTO, False, True, DeviceKind.CUDA, False),
        (GpuMode.AUTO, False, False, DeviceKind.CPU, False),
        (GpuMode.AUTO, True, True, DeviceKind.CPU, False),
        (GpuMode.ON, False, True, DeviceKind.CUDA, False),
        (GpuMode.ON, True, False, DeviceKind.CPU, False),
        (GpuMode.ON, False, False, DeviceKind.CPU, True),
    ],
)
def test_device_manager_matrix(
    tmp_path, gpu_mode, paused, cuda_available, expect_kind, expect_error
) -> None:
    env = _env(tmp_path, gpu_mode)
    pause = FakePause(paused)
    manager = DeviceManager(env, pause_controller=pause, cuda_available_fn=lambda: cuda_available)
    if expect_error:
        with pytest.raises(RuntimeError):
            manager.select_device()
        return
    selection = manager.select_device()
    assert selection.device_kind == expect_kind
