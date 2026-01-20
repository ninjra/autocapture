"""Embedding and reranking providers for Memory Service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from ..logging_utils import get_logger
from ..text.normalize import normalize_text
from ..config import MemoryServiceEmbedderConfig, MemoryServiceRerankerConfig
from .utils import hash_embedding

_LOG = get_logger("memory.providers")


class Embedder(Protocol):
    dim: int
    model_id: str

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: Iterable[str]) -> list[float]: ...


@dataclass
class HashEmbedder:
    dim: int
    model_id: str = "hash-v1"

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return [hash_embedding(text or "", self.dim) for text in texts]


class EmbeddingServiceAdapter:
    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self.dim = int(getattr(embedder, "dim", 0) or 0)
        self.model_id = getattr(embedder, "model_name", "local")

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return self._embedder.embed_texts(texts)


@dataclass
class StubReranker:
    def score(self, query: str, texts: Iterable[str]) -> list[float]:
        query_terms = _tokenize(query)
        scores: list[float] = []
        for text in texts:
            terms = _tokenize(text)
            if not terms:
                scores.append(0.0)
                continue
            overlap = len(query_terms & terms)
            scores.append(overlap / max(len(terms), 1))
        return scores


def build_embedder(config: MemoryServiceEmbedderConfig, *, allow_local: bool):
    provider = (config.provider or "stub").strip().lower()
    if provider == "local" and allow_local:
        try:
            from ..embeddings.service import EmbeddingService
            from ..config import EmbedConfig

            service = EmbeddingService(EmbedConfig())
            return EmbeddingServiceAdapter(service)
        except Exception as exc:
            _LOG.warning("Local embedder unavailable; using stub ({})", exc)
            return HashEmbedder(dim=config.dim, model_id=config.model_id)
    return HashEmbedder(dim=config.dim, model_id=config.model_id)


def build_reranker(config: MemoryServiceRerankerConfig):
    provider = (config.provider or "disabled").strip().lower()
    if provider == "stub":
        return StubReranker()
    return None


def _tokenize(text: str) -> set[str]:
    normalized = normalize_text(text).lower()
    return {token for token in normalized.split(" ") if token}
