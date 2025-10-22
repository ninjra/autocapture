"""Embedding batchers and helpers.

The heavy pipeline implementation is imported lazily so running
``python -m autocapture.embeddings.pipeline`` does not produce duplicate
import warnings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - only for static analyzers
    from .pipeline import EmbeddingBatcher

__all__ = ["EmbeddingBatcher"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
