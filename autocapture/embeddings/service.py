"""Embedding service wrapper with fastembed + sentence-transformers fallback."""

from __future__ import annotations

from typing import Iterable

from ..config import EmbedConfig, is_dev_mode
from ..logging_utils import get_logger
from ..gpu_lease import get_global_gpu_lease
from ..runtime_pause import PauseController, paused_guard


class EmbeddingService:
    def __init__(
        self, config: EmbedConfig, *, pause_controller: PauseController | None = None
    ) -> None:
        self._config = config
        self._log = get_logger("embeddings")
        self._backend = None
        self._dim = None
        self._model_name = config.text_model
        self._lease_key = f"embed:{id(self)}"
        self._pause = pause_controller
        get_global_gpu_lease().register_release_hook(self._lease_key, self._on_release)
        self._init_backend()

    def _init_backend(self) -> None:
        paused_guard(self._pause)
        if self._model_name == "local-test":
            self._backend = "local-test"
            self._dim = 16
            self._log.info("Embedding backend: local-test")
            return
        import importlib.util

        if importlib.util.find_spec("fastembed") is not None:
            from fastembed import TextEmbedding  # type: ignore

            self._backend = TextEmbedding(model_name=self._model_name)
            self._dim = self._backend.embedding_size
            self._log.info("Embedding backend: fastembed ({})", self._model_name)
            return

        if importlib.util.find_spec("sentence_transformers") is not None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._backend = SentenceTransformer(self._model_name)
            self._dim = self._backend.get_sentence_embedding_dimension()
            self._log.info("Embedding backend: sentence-transformers ({})", self._model_name)
            return

        if is_dev_mode():
            self._backend = "dev-fallback"
            self._dim = 16
            self._log.warning("Embedding backend: dev-fallback (hash) for {}", self._model_name)
            return

        raise RuntimeError("No embedding backend available")

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Embedding backend not initialized")
        return int(self._dim)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = [text or "" for text in texts]
        if not text_list:
            return []
        if self._backend is None:
            self._init_backend()
        if self._backend is None:
            raise RuntimeError("Embedding backend not initialized")

        if self._backend in {"local-test", "dev-fallback"}:
            return [_hash_embedding(text, self.dim) for text in text_list]

        if hasattr(self._backend, "embed"):
            vectors = list(self._backend.embed(text_list))
            return [
                vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors
            ]

        vectors = self._backend.encode(
            text_list,
            batch_size=self._config.text_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [vec.tolist() for vec in vectors]

    def close(self) -> None:
        if self._backend and self._backend not in {"local-test", "dev-fallback"}:
            _try_empty_cuda_cache()
        self._backend = None

    def _on_release(self, reason: str) -> None:
        _ = reason
        self.close()


def _hash_embedding(text: str, dim: int) -> list[float]:
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [b / 255.0 for b in digest[:dim]]
    if len(values) < dim:
        values.extend([0.0] * (dim - len(values)))
    return values


def _try_empty_cuda_cache() -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
