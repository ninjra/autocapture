"""RapidOCR extraction wrapper."""

from __future__ import annotations

import importlib.util
from typing import Iterable

import numpy as np

from ..config import OCRConfig
from ..logging_utils import get_logger


def available_onnx_providers() -> list[str]:
    if importlib.util.find_spec("onnxruntime") is None:
        return []
    import onnxruntime as ort  # type: ignore

    return list(ort.get_available_providers())


def select_onnx_provider(config: OCRConfig, providers: Iterable[str]) -> tuple[str | None, bool]:
    preferred = list(config.onnx_providers or [])
    device = config.device.lower()
    if device == "cpu":
        preferred = ["CPUExecutionProvider"]
    elif device == "cuda" and not preferred:
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if not preferred:
        preferred = ["CPUExecutionProvider"]
    for provider in preferred:
        if provider in providers:
            return provider, provider == "CUDAExecutionProvider"
    return None, False


def _build_rapidocr_kwargs(use_cuda: bool) -> dict[str, object]:
    return {"use_cuda": use_cuda}


class RapidOCRExtractor:
    def __init__(self, config: OCRConfig) -> None:
        if importlib.util.find_spec("rapidocr_onnxruntime") is None:
            raise RuntimeError("rapidocr_onnxruntime is required for RapidOCR extraction")
        from rapidocr_onnxruntime import RapidOCR

        self._config = config
        self._log = get_logger("ocr")
        providers = available_onnx_providers()
        self._log.info("ONNX Runtime providers available: {}", providers or "none")
        selected, use_cuda = select_onnx_provider(config, providers)
        if config.device.lower() == "cuda" and selected != "CUDAExecutionProvider":
            self._log.warning(
                "OCR device=cuda but CUDAExecutionProvider unavailable; falling back to CPU. "
                "Install onnxruntime-gpu and CUDA/cuDNN."
            )
        selected_name = selected or "none"
        self._log.info("OCR execution provider selected: {}", selected_name)
        kwargs = _build_rapidocr_kwargs(use_cuda)
        if kwargs:
            self._log.info("RapidOCR init kwargs: {}", kwargs)
        self._engine = RapidOCR(**kwargs)
        self._warmup()

    def _warmup(self) -> None:
        sample = np.zeros((16, 16, 3), dtype=np.uint8)
        self._engine(sample)

    def extract(self, image: np.ndarray) -> list[tuple[str, float, list[int]]]:
        bgr = image[:, :, ::-1]
        results, _ = self._engine(bgr)
        spans = []
        for result in results or []:
            box, text, confidence = result
            flattened = [int(coord) for point in box for coord in point]
            spans.append((text, float(confidence), flattened))
        return spans
