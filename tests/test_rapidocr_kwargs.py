from __future__ import annotations

from autocapture.config import OCRConfig
from autocapture.worker.event_worker import _build_rapidocr_kwargs, _select_onnx_provider


def test_build_rapidocr_kwargs_sets_use_cuda() -> None:
    assert _build_rapidocr_kwargs(True) == {"use_cuda": True}
    assert _build_rapidocr_kwargs(False) == {"use_cuda": False}


def test_select_onnx_provider_prefers_cuda_then_cpu() -> None:
    config = OCRConfig(
        device="cuda", onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    provider, use_cuda = _select_onnx_provider(
        config,
        ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    assert provider == "CUDAExecutionProvider"
    assert use_cuda is True


def test_select_onnx_provider_honors_cpu_device() -> None:
    config = OCRConfig(
        device="cpu", onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    provider, use_cuda = _select_onnx_provider(config, ["CPUExecutionProvider"])
    assert provider == "CPUExecutionProvider"
    assert use_cuda is False
