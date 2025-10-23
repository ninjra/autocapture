"""OCR pipeline with GPU-first processing."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image

from ..config import OCRConfig
from ..logging_utils import get_logger

if hasattr(mp, "set_executable"):
    mp.set_executable(sys.executable)


@dataclass(slots=True)
class OCRJob:
    image_path: Path
    capture_timestamp: str
    foreground_process: str
    foreground_window: str
    monitor_id: str


@dataclass(slots=True)
class OCRSpan:
    text: str
    confidence: float
    bbox: list[int]


@dataclass(slots=True)
class OCRResult:
    job: OCRJob
    spans: list[OCRSpan]
    raw: dict


class OCRWorker:
    """Async worker that batches OCR jobs and emits structured spans."""

    def __init__(
        self, config: OCRConfig, on_result: Callable[[OCRResult], None]
    ) -> None:
        self._config = config
        self._on_result = on_result
        self._queue: asyncio.Queue[OCRJob] = asyncio.Queue(maxsize=config.queue_maxsize)
        self._log = get_logger("ocr")
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        self._log.info("OCR worker started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._log.info("OCR worker stopped")

    async def submit(self, job: OCRJob) -> None:
        await self._queue.put(job)

    async def _run(self) -> None:
        batch: list[OCRJob] = []
        while True:
            try:
                job = await asyncio.wait_for(
                    self._queue.get(), timeout=self._config.max_latency_s
                )
            except asyncio.TimeoutError:
                job = None

            if job:
                batch.append(job)

            if not batch:
                continue

            if len(batch) >= self._config.batch_size or job is None:
                await self._process_batch(batch)
                batch = []

    async def _process_batch(self, batch: list[OCRJob]) -> None:
        images = [Image.open(job.image_path) for job in batch]
        results = self._run_engine(images)
        for job, result in zip(batch, results, strict=True):
            spans = [
                OCRSpan(
                    text=span["text"], confidence=span["confidence"], bbox=span["bbox"]
                )
                for span in result.get("spans", [])
            ]
            ocr_result = OCRResult(job=job, spans=spans, raw=result)
            try:
                self._on_result(ocr_result)
            except Exception as exc:  # pragma: no cover
                self._log.exception(
                    "Failed to dispatch OCR result for %s: %s", job.image_path, exc
                )

    def _run_engine(self, images: Iterable[Image.Image]) -> list[dict]:
        """Invoke PaddleOCR/EasyOCR depending on configuration."""

        engine = self._config.engine
        if engine.startswith("paddleocr"):
            return self._run_paddle(images)
        if engine.startswith("easyocr"):
            return self._run_easyocr(images)
        raise ValueError(f"Unsupported OCR engine: {engine}")

    def _run_paddle(
        self, images: Iterable[Image.Image]
    ) -> list[dict]:  # pragma: no cover
        from paddleocr import PaddleOCR

        import numpy as np

        ocr = PaddleOCR(
            use_angle_cls=True, use_gpu=True, lang="+".join(self._config.languages)
        )
        results = []
        for image in images:
            raw = ocr.ocr(np.array(image), cls=True)
            spans = [
                {
                    "text": line[1][0],
                    "confidence": float(line[1][1]),
                    "bbox": [int(coord) for point in line[0] for coord in point],
                }
                for line in raw[0]
            ]
            results.append({"spans": spans, "engine": "paddleocr"})
        return results

    def _run_easyocr(
        self, images: Iterable[Image.Image]
    ) -> list[dict]:  # pragma: no cover
        import easyocr
        import numpy as np

        reader = easyocr.Reader(self._config.languages, gpu=True)
        results = []
        for image in images:
            raw = reader.readtext(np.array(image), detail=1)
            spans = [
                {
                    "text": line[1],
                    "confidence": float(line[2]),
                    "bbox": [int(coord) for point in line[0] for coord in point],
                }
                for line in raw
            ]
            results.append({"spans": spans, "engine": "easyocr"})
        return results
