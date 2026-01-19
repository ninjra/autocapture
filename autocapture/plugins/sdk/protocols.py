"""Protocols for plugin implementations."""

from __future__ import annotations

from typing import Protocol, Sequence

from ...llm.providers import LLMProvider
from ...vision.types import ExtractionResult
from ...indexing.vector_index import VectorBackend
from ...embeddings.service import EmbeddingService
from ...memory.reranker import CrossEncoderReranker
from ...memory.compression import CompressedAnswer
from ...memory.context_pack import EvidenceItem
from ...memory.verification import Claim


class VisionExtractor(Protocol):
    def extract(self, image) -> ExtractionResult: ...


class Embedder(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def rank(self, query: str, documents: Sequence[str], *, batch_size: int | None = None): ...


class Compressor(Protocol):
    def compress(self, evidence: list[EvidenceItem]) -> CompressedAnswer: ...


class Verifier(Protocol):
    def verify(
        self, claims: list[Claim], valid_evidence: set[str], entity_tokens: set[str]
    ) -> list[str]: ...


class ResearchSource(Protocol):
    def fetch(self, **kwargs): ...


__all__ = [
    "LLMProvider",
    "VisionExtractor",
    "VectorBackend",
    "EmbeddingService",
    "CrossEncoderReranker",
    "Embedder",
    "Reranker",
    "Compressor",
    "Verifier",
    "ResearchSource",
]
