"""Local FastAPI server for Personal Activity Memory Engine."""

from __future__ import annotations

from fastapi import FastAPI

from ..config import AppConfig
from ..embeddings.service import EmbeddingService
from ..indexing.vector_index import VectorIndex
from ..plugins import PluginManager
from ..storage.database import DatabaseManager
from .app import create_app as build_app
from .container import build_container
from .routers.core import (
    RetrieveRequest,
    _remap_spans,
    _resolve_retrieve_paging,
    _snippet_for_query,
    _spans_for_event,
)

__all__ = [
    "create_app",
    "RetrieveRequest",
    "_resolve_retrieve_paging",
    "_remap_spans",
    "_snippet_for_query",
    "_spans_for_event",
]


def create_app(
    config: AppConfig,
    db_manager: DatabaseManager | None = None,
    *,
    embedder: EmbeddingService | None = None,
    vector_index: VectorIndex | None = None,
    worker_supervisor: object | None = None,
    plugin_manager: PluginManager | None = None,
) -> FastAPI:
    container = build_container(
        config,
        db_manager=db_manager,
        embedder=embedder,
        vector_index=vector_index,
        worker_supervisor=worker_supervisor,
        plugin_manager=plugin_manager,
    )
    return build_app(container)
