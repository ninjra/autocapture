from __future__ import annotations

import datetime as dt

from autocapture.config import AppConfig, DatabaseConfig, QdrantConfig
from autocapture.memory.graph_adapters import GraphHit
from autocapture.memory.retrieval import RetrievalService
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord


class _StubEmbedder:
    model_name = "stub"
    dim = 3

    def embed_texts(self, texts):
        return [[0.1, 0.1, 0.1] for _ in texts]


class _StubVectorIndex:
    def search(self, *_args, **_kwargs):
        return []


class _StubGraphAdapters:
    def enabled(self) -> bool:
        return True

    def query(self, _query: str, *, limit: int, time_range, filters):
        _ = limit, time_range, filters
        return [GraphHit(event_id="e1", score=0.9, snippet="hit", source="graphrag")]


def test_retrieval_uses_graph_adapters_when_available() -> None:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False),
        qdrant=QdrantConfig(enabled=False),
    )
    config.routing.reranker = "disabled"
    config.reranker.enabled = False
    db = DatabaseManager(config.database)
    now = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.timezone.utc)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="e1",
                ts_start=now,
                ts_end=None,
                app_name="App",
                window_title="Title",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="irrelevant",
                embedding_vector=None,
                embedding_status="pending",
                embedding_model=None,
                tags={},
            )
        )
    service = RetrievalService(
        db,
        config,
        embedder=_StubEmbedder(),
        vector_index=_StubVectorIndex(),
        graph_adapters=_StubGraphAdapters(),
    )
    batch = service.retrieve("missing", None, None, limit=1)
    assert batch.results
    assert batch.results[0].event.event_id == "e1"
