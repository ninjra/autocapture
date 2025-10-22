"""OCR worker entry points.

The concrete classes live in :mod:`autocapture.ocr.pipeline`. Importing
them lazily avoids double-import warnings when the module is executed as a
script via ``python -m autocapture.ocr.pipeline``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - only for static checkers
    from .pipeline import OCRJob, OCRResult, OCRWorker

__all__ = ["OCRWorker", "OCRJob", "OCRResult"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
