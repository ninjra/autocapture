"""Constants for plugin system."""

from __future__ import annotations

EXTENSION_KINDS = (
    "llm.provider",
    "vision.extractor",
    "ocr.engine",
    "embedder.text",
    "retrieval.strategy",
    "vector.backend",
    "reranker",
    "compressor",
    "verifier",
    "research.source",
    "research.watchlist",
    "agent.job",
    "prompt.bundle",
    "ui.panel",
    "ui.overlay",
)

FACTORY_TYPES = ("python", "bundle", "file")
