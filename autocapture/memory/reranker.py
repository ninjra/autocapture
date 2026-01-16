"""Cross-encoder reranker wrapper."""

from __future__ import annotations

import importlib.util
from typing import Sequence

from ..config import RerankerConfig
from ..logging_utils import get_logger


class CrossEncoderReranker:
    def __init__(self, config: RerankerConfig) -> None:
        self._config = config
        self._log = get_logger("reranker")
        self._device = _resolve_device(config.device)
        self._backend: object | None = None

        if config.model == "local-test":
            self._backend = "local-test"
            self._log.info("Reranker backend: local-test")
            return

        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError("sentence-transformers is required for reranking")

        from sentence_transformers import CrossEncoder  # type: ignore

        self._backend = CrossEncoder(config.model, device=self._device)
        self._log.info(
            "Reranker backend: sentence-transformers ({}) on {}",
            config.model,
            self._device,
        )

    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        docs = [doc or "" for doc in documents]
        if not docs:
            return []
        if self._backend is None:
            raise RuntimeError("Reranker backend not initialized")
        if self._backend == "local-test":
            return _hash_scores(query, docs)

        pairs = [(query, doc) for doc in docs]
        scores = self._backend.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return [float(score) for score in scores]


def _resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if normalized == "cuda" and not _cuda_available():
        return "cpu"
    return normalized


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch  # type: ignore

    return bool(torch.cuda.is_available())


def _hash_scores(query: str, documents: Sequence[str]) -> list[float]:
    import hashlib

    scores: list[float] = []
    for doc in documents:
        digest = hashlib.sha256(f"{query}:{doc}".encode("utf-8")).digest()
        scores.append(digest[0] / 255.0)
    return scores
