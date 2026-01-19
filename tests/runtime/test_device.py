from __future__ import annotations

from pathlib import Path
import re

import pytest

from autocapture.runtime_device import DeviceManager, DeviceKind, GpuRequiredError
from autocapture.runtime_env import GpuMode, ProfileName, RuntimeEnvConfig


def _env(tmp_path: Path, gpu_mode: GpuMode) -> RuntimeEnvConfig:
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
    "gpu_mode,cuda_available,expect_kind,expect_error",
    [
        (GpuMode.OFF, False, DeviceKind.CPU, False),
        (GpuMode.OFF, True, DeviceKind.CPU, False),
        (GpuMode.AUTO, True, DeviceKind.CUDA, False),
        (GpuMode.AUTO, False, DeviceKind.CPU, False),
        (GpuMode.ON, True, DeviceKind.CUDA, False),
        (GpuMode.ON, False, DeviceKind.CPU, True),
    ],
)
def test_device_manager_matrix(
    tmp_path, gpu_mode, cuda_available, expect_kind, expect_error
) -> None:
    env = _env(tmp_path, gpu_mode)
    manager = DeviceManager(
        env, cuda_detect_fn=lambda: (cuda_available, {"cuda_available": cuda_available})
    )
    if expect_error:
        with pytest.raises(GpuRequiredError):
            manager.select_device()
        return
    selection = manager.select_device()
    assert selection.device_kind == expect_kind


def test_no_direct_torch_device_cuda() -> None:
    pattern = re.compile(r"torch\.device\(\s*['\"]cuda", re.IGNORECASE)
    offenders: list[str] = []
    for path in Path("autocapture").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for idx, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path}:{idx}:{line.strip()}")
    if offenders:
        joined = "\n".join(offenders)
        raise AssertionError(f"Direct torch.device('cuda') usage found:\n{joined}")
