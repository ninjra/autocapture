from __future__ import annotations

import importlib.util
import threading
import time

import pytest

from autocapture.runtime_pause import PauseController


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch  # type: ignore

    return bool(torch.cuda.is_available())


@pytest.mark.gpu
def test_gpu_kernel_parity() -> None:
    if not _cuda_available():
        pytest.skip("CUDA unavailable")
    import torch  # type: ignore

    torch.manual_seed(0)
    a = torch.randn((128, 128))
    b = torch.randn((128, 128))
    cpu = torch.matmul(a, b)
    gpu = torch.matmul(a.cuda(), b.cuda()).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-3, rtol=1e-3)


@pytest.mark.gpu
def test_no_gpu_work_occurs_when_paused(tmp_path) -> None:
    if not _cuda_available():
        pytest.skip("CUDA unavailable")
    import torch  # type: ignore

    pause = PauseController(
        tmp_path / "pause.flag",
        tmp_path / "pause_reason.json",
        poll_interval_s=0.01,
    )
    pause.pause("manual", "unit")

    entered = threading.Event()
    done = threading.Event()

    def _worker() -> None:
        pause.wait_until_resumed()
        x = torch.randn((64, 64), device="cuda")
        y = torch.matmul(x, x)
        _ = y.cpu()
        torch.cuda.synchronize()
        entered.set()
        done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    time.sleep(0.1)
    assert entered.is_set() is False

    pause.resume("unit")
    done.wait(timeout=2.0)
    assert entered.is_set() is True
