"""Stable SDK surface for plugin authors."""

from .context import (
    PluginContext,
    LLMProviderContext,
    LLMProviderInfo,
    VisionExtractorContext,
    RetrievalContext,
)
from .protocols import (
    LLMProvider,
    VisionExtractor,
    VectorBackend,
    EmbeddingService,
    CrossEncoderReranker,
    Embedder,
    Reranker,
    Compressor,
    Verifier,
    ResearchSource,
    GraphAdapter,
    DecodeBackend,
    TrainingPipeline,
)

__all__ = [
    "PluginContext",
    "LLMProviderContext",
    "LLMProviderInfo",
    "VisionExtractorContext",
    "RetrievalContext",
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
]
