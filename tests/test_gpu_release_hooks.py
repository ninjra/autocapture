from __future__ import annotations

import time

import pytest

from autocapture.config import RuntimeConfig
from autocapture.gpu_lease import GpuLease
from autocapture.runtime_governor import FullscreenState, RuntimeGovernor, WindowMonitor


class FakeRawInput:
    def __init__(self) -> None:
        self.last_input_ts = int(time.monotonic() * 1000)


class FakeWindowMonitor(WindowMonitor):
    def __init__(self) -> None:
        super().__init__()
        self.state = FullscreenState(False, None, None, None)

    def sample(self) -> FullscreenState:
        return self.state


def test_gpu_release_hooks_fire_once_per_transition() -> None:
    config = RuntimeConfig()
    raw = FakeRawInput()
    monitor = FakeWindowMonitor()
    lease = GpuLease()
    calls: list[str] = []
    lease.register_release_hook("test", lambda reason: calls.append(reason))
    governor = RuntimeGovernor(
        config,
        raw_input=raw,
        window_monitor=monitor,
        gpu_lease=lease,
    )

    monitor.state = FullscreenState(True, 1, "app.exe", "Full Screen")
    governor.tick()
    governor.tick()
    assert calls == ["fullscreen"]

    monitor.state = FullscreenState(False, None, None, None)
    governor.tick()
    monitor.state = FullscreenState(True, 1, "app.exe", "Full Screen")
    governor.tick()
    assert calls.count("fullscreen") == 2


def test_gpu_release_nvml_optional() -> None:
    pynvml = pytest.importorskip("pynvml", reason="NVML bindings not installed")
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"NVML init failed: {exc}")
    try:
        if pynvml.nvmlDeviceGetCount() <= 0:
            pytest.skip("No NVML GPUs available")
        try:
            import torch  # type: ignore
        except Exception:
            pytest.skip("Torch not available for GPU allocation")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        tensor_holder = {"tensor": torch.zeros((1024, 1024, 64), device="cuda")}
        mid = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        if mid <= before:
            pytest.skip("Unable to observe GPU allocation delta")

        lease = GpuLease()

        def _release(_reason: str) -> None:
            tensor_holder["tensor"] = None
            torch.cuda.empty_cache()

        lease.register_release_hook("nvml", _release)
        lease.release_all("test")
        after = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        assert after <= mid
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
