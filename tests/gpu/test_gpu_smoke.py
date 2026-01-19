from __future__ import annotations

import threading
import time

import pytest

from autocapture.runtime_device import DeviceManager
from autocapture.runtime_env import load_runtime_env
from autocapture.runtime_pause import PauseController


def _torch_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.gpu
def test_gpu_smoke_pause_gate(monkeypatch, tmp_path) -> None:
    if not _torch_available():
        pytest.skip("CUDA unavailable or torch missing")

    monkeypatch.setenv("AUTOCAPTURE_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOCAPTURE_GPU_MODE", "auto")
    runtime_env = load_runtime_env()
    pause = PauseController(runtime_env, poll_interval_s=0.01)
    device_manager = DeviceManager(runtime_env, pause_controller=pause)

    compute_device, _ = device_manager.resolve_compute_device()
    assert compute_device == "cuda"

    import torch  # type: ignore

    selection = device_manager.select_device()
    device = torch.device(selection.torch_device or selection.compute_device)

    pause.write_pause("manual", "unit")
    counter = {"value": 0}
    stop_event = threading.Event()

    def _worker() -> None:
        while not stop_event.is_set():
            if pause.is_paused():
                pause.wait_until_resumed(timeout=0.1)
                continue
            x = torch.randn((16, 16), device=device)
            y = torch.matmul(x, x)
            _ = y.cpu()
            torch.cuda.synchronize()
            counter["value"] += 1
            stop_event.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    time.sleep(0.2)
    assert counter["value"] == 0

    pause.clear_pause("unit")
    stop_event.wait(timeout=2.0)
    assert counter["value"] >= 1
