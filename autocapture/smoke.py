"""Deterministic smoke checks for config, DB, and embeddings."""

from __future__ import annotations

import time

from sqlalchemy import text

from .config import AppConfig
from .doctor import _check_migrations
from .embeddings.service import EmbeddingService
from .logging_utils import get_logger
from .storage.database import DatabaseManager


def run_smoke(config: AppConfig) -> int:
    log = get_logger("smoke")
    started = time.monotonic()
    try:
        db = DatabaseManager(config.database)
        with db.session() as session:
            session.execute(text("SELECT 1"))
        _check_migrations(db)
        embedder = EmbeddingService(config.embed)
        vectors = embedder.embed_texts(["smoke"])
        if not vectors or not vectors[0]:
            log.error("Smoke failed: embedder returned empty vectors.")
            return 2
    except Exception as exc:
        log.error("Smoke failed: {}", exc)
        return 2
    elapsed_ms = (time.monotonic() - started) * 1000
    log.info(
        "Smoke passed in {:.0f}ms (embedder={}, dim={})",
        elapsed_ms,
        embedder.model_name,
        embedder.dim,
    )
    return 0
