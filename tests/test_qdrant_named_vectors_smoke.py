from __future__ import annotations

from dataclasses import dataclass

from autocapture.config import AppConfig
from autocapture.indexing.spans_v2 import QdrantSpansV2Backend, SpanV2Upsert, SparseEmbedding


def _qdrant_available(url: str) -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url, timeout=1.0)
        client.get_collections()
        return True
    except Exception:
        return False


def _sample_upsert(dim: int) -> SpanV2Upsert:
    return SpanV2Upsert(
        capture_id="event-1",
        span_key="S1",
        dense_vector=[0.01] * dim,
        sparse_vector=SparseEmbedding(indices=[1, 2], values=[0.2, 0.1]),
        late_vectors=[[0.02] * dim],
        payload={"text": "hello"},
        embedding_model="test-model",
    )


def test_qdrant_named_vectors_smoke_or_contract(monkeypatch) -> None:
    config = AppConfig()
    config.qdrant.spans_v2_collection = "spans_v2_test"
    dim = 8

    if _qdrant_available(config.qdrant.url):
        from autocapture.indexing.spans_v2 import SpansV2Index

        index = SpansV2Index(config, dim)
        index.upsert([_sample_upsert(dim)])
        hits = index.search_dense([0.01] * dim, 1, embedding_model="test-model")
        assert hits
        return

    @dataclass
    class _FakeClient:
        points: list = None

        def upsert(self, _collection_name, points):
            self.points = points

    backend = QdrantSpansV2Backend(config, dim)
    fake = _FakeClient()
    backend._client = fake  # type: ignore[assignment]
    monkeypatch.setattr(backend, "_ensure_collection", lambda: None)
    backend.upsert([_sample_upsert(dim)])
    assert fake.points is not None
    point = fake.points[0]
    vector = getattr(point, "vector", {})
    assert "dense" in vector
    assert "late" in vector
    sparse_vectors = getattr(point, "sparse_vectors", None)
    if sparse_vectors:
        assert "sparse" in sparse_vectors
    else:
        assert "sparse" in vector
    payload = getattr(point, "payload", {})
    assert payload.get("capture_id") == "event-1"
    assert payload.get("span_key") == "S1"
    assert payload.get("embedding_model") == "test-model"
