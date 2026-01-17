"""Cross-encoder reranker wrapper."""

from __future__ import annotations

import importlib.util
from typing import Sequence

from ..config import RerankerConfig
from ..logging_utils import get_logger
from ..gpu_lease import get_global_gpu_lease


class CrossEncoderReranker:
    def __init__(self, config: RerankerConfig) -> None:
        self._config = config
        self._log = get_logger("reranker")
        self._device = _resolve_device(config.device)
        self._current_device = self._device
        self._backend: object | None = None
        self._lease_key = f"reranker:{id(self)}"
        get_global_gpu_lease().register_release_hook(self._lease_key, self._on_release)

        self._init_backend(self._device)

    def rank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> list[float]:
        docs = [doc or "" for doc in documents]
        if not docs:
            return []
        if device:
            self.ensure_device(device)
        if self._backend is None:
            self._init_backend(self._device)
        if self._backend is None:
            raise RuntimeError("Reranker backend not initialized")
        if self._backend == "local-test":
            return _hash_scores(query, docs)

        pairs = [(query, doc) for doc in docs]
        scores: list[float] = []
        if batch_size is None or batch_size <= 0:
            batch_scores = self._backend.predict(pairs)
            if hasattr(batch_scores, "tolist"):
                batch_scores = batch_scores.tolist()
            return [float(score) for score in batch_scores]
        for idx in range(0, len(pairs), batch_size):
            batch = pairs[idx : idx + batch_size]
            batch_scores = self._backend.predict(batch)
            if hasattr(batch_scores, "tolist"):
                batch_scores = batch_scores.tolist()
            scores.extend([float(score) for score in batch_scores])
        return scores

    def ensure_device(self, device: str) -> None:
        resolved = _resolve_device(device)
        if resolved == self._current_device and self._backend is not None:
            return
        self._device = resolved
        self._init_backend(resolved)

    def close(self) -> None:
        if self._backend and self._backend != "local-test":
            _try_empty_cuda_cache()
        self._backend = None

    def _on_release(self, reason: str) -> None:
        _ = reason
        self.close()

    def _init_backend(self, device: str) -> None:
        if self._config.model == "local-test":
            self._backend = "local-test"
            self._current_device = device
            self._log.info("Reranker backend: local-test")
            return
        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError("sentence-transformers is required for reranking")
        from sentence_transformers import CrossEncoder  # type: ignore

        self._backend = CrossEncoder(self._config.model, device=device)
        self._current_device = device
        self._log.info(
            "Reranker backend: sentence-transformers ({}) on {}",
            self._config.model,
            device,
        )


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


def _try_empty_cuda_cache() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _hash_scores(query: str, documents: Sequence[str]) -> list[float]:
    import hashlib

    scores: list[float] = []
    for doc in documents:
        digest = hashlib.sha256(f"{query}:{doc}".encode("utf-8")).digest()
        scores.append(digest[0] / 255.0)
    return scores
