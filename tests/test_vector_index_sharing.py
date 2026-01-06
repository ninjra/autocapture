from __future__ import annotations

from pathlib import Path

from autocapture.config import AppConfig, CaptureConfig, DatabaseConfig, WorkerConfig
from autocapture.indexing.vector_index import (
    SpanEmbeddingUpsert,
    VectorHit,
    VectorIndex,
)
from autocapture.memory.retrieval import RetrievalService
from autocapture.storage.database import DatabaseManager
from autocapture.worker.embedding_worker import EmbeddingWorker


class DummyEmbedder:
    def __init__(self, dim: int = 3) -> None:
        self.model_name = "dummy"
        self.dim = dim

    def embed_texts(self, texts):
        return [[0.1] * self.dim for _ in texts]


class FakeBackend:
    def __init__(self) -> None:
        self.upserts: list[SpanEmbeddingUpsert] = []

    def upsert_spans(self, upserts: list[SpanEmbeddingUpsert]) -> None:
        self.upserts.extend(upserts)

    def search(self, query_vector, k, *, filters=None, embedding_model: str):
        return [
            VectorHit(event_id=up.capture_id, span_key=up.span_key, score=0.9)
            for up in self.upserts
            if up.embedding_model == embedding_model
        ][:k]


def test_vector_index_shared_instance(tmp_path: Path) -> None:
    config = AppConfig(
        capture=CaptureConfig(staging_dir=tmp_path / "staging", data_dir=tmp_path),
        worker=WorkerConfig(data_dir=tmp_path),
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
    )
    db = DatabaseManager(config.database)
    dim = 3

    backend = FakeBackend()
    vector_index = VectorIndex(config, dim, backend=backend)
    embedder = DummyEmbedder(dim=dim)
    worker = EmbeddingWorker(
        config,
        db_manager=db,
        embedder=embedder,
        vector_index=vector_index,
    )
    retrieval = RetrievalService(
        db,
        config,
        embedder=embedder,
        vector_index=vector_index,
    )

    assert worker._vector_index is vector_index
    assert retrieval._vector is vector_index
