"""Protocols for plugin implementations."""

from __future__ import annotations

from typing import Protocol, Sequence, TYPE_CHECKING

from ...llm.providers import LLMProvider
from ...vision.types import ExtractionResult
from ...indexing.vector_index import VectorBackend
from ...embeddings.service import EmbeddingService
from ...memory.reranker import CrossEncoderReranker
from ...memory.compression import CompressedAnswer
from ...memory.context_pack import EvidenceItem
from ...config import CircuitBreakerConfig
from ...memory.graph_adapters import GraphHit
from ...training.models import TrainingRunRequest, TrainingRunResult
from ...memory.verification import Claim

if TYPE_CHECKING:
    from ...enrichment.table_extractor import (
        TableExtractionRequest,
        TableExtractionResult,
        TableExtractorSpec,
    )


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


class GraphAdapter(Protocol):
    @property
    def enabled(self) -> bool: ...

    def query(
        self,
        query: str,
        *,
        limit: int,
        time_range: tuple[str, str] | None,
        filters: dict | None,
    ) -> list[GraphHit]: ...


class DecodeBackend(Protocol):
    id: str
    type: str
    base_url: str | None
    api_key_env: str | None
    api_key: str | None
    timeout_s: float
    retries: int
    headers: dict[str, str]
    allow_cloud: bool
    circuit_breaker: CircuitBreakerConfig
    max_concurrency: int


class TrainingPipeline(Protocol):
    def run(self, request: TrainingRunRequest) -> TrainingRunResult: ...


class TableExtractor(Protocol):
    def describe(self) -> "TableExtractorSpec": ...

    def extract(self, request: "TableExtractionRequest") -> "TableExtractionResult": ...


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
    "GraphAdapter",
    "DecodeBackend",
    "TrainingPipeline",
    "TableExtractor",
]
