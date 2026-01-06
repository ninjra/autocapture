from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from autocapture.config import AppConfig, CaptureConfig, DatabaseConfig, WorkerConfig
from autocapture.indexing.vector_index import HNSWIndex, VectorIndex
from autocapture.memory.retrieval import RetrievalService
from autocapture.storage.database import DatabaseManager
from autocapture.worker.embedding_worker import EmbeddingWorker


class DummyEmbedder:
    def __init__(self, dim: int = 3) -> None:
        self.model_name = "dummy"
        self.dim = dim

    def embed_texts(self, texts):
        return [[0.1] * self.dim for _ in texts]


def test_vector_index_shared_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = AppConfig(
        capture=CaptureConfig(staging_dir=tmp_path / "staging", data_dir=tmp_path),
        worker=WorkerConfig(data_dir=tmp_path),
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
    )
    db = DatabaseManager(config.database)
    dim = 3

    def _select_backend(self):
        self._backend = HNSWIndex(config, db, dim)

    monkeypatch.setattr(VectorIndex, "_select_backend", _select_backend)

    vector_index = VectorIndex(config, db, dim)
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


def test_hnsw_index_reload(tmp_path: Path) -> None:
    pytest.importorskip("hnswlib")
    config = AppConfig(
        capture=CaptureConfig(staging_dir=tmp_path / "staging", data_dir=tmp_path),
        worker=WorkerConfig(data_dir=tmp_path),
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
    )
    db = DatabaseManager(config.database)
    index_a = HNSWIndex(config, db, dim=3)
    index_b = HNSWIndex(config, db, dim=3)
    index_a.upsert_spans(
        [
            (
                "event-1",
                "S1",
                [0.1, 0.2, 0.3],
                {
                    "span_id": 1,
                    "ts_start": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "event_id": "event-1",
                    "span_key": "S1",
                },
            )
        ]
    )
    hits = index_b.search([0.1, 0.2, 0.3], limit=1)
    assert hits
    assert hits[0].event_id == "event-1"
