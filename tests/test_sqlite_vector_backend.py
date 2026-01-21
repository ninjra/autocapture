from __future__ import annotations

from pathlib import Path

import pytest

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.indexing.sqlite_backends import SqliteVectorBackend
from autocapture.indexing.sqlite_utils import cosine_similarity, vector_norm
from autocapture.indexing.vector_index import SpanEmbeddingUpsert
from autocapture.storage.database import DatabaseManager


def _db_config(path: Path, *, encrypted: bool) -> DatabaseConfig:
    cfg = DatabaseConfig(url=f"sqlite:///{path}", sqlite_wal=False)
    if encrypted:
        cfg.encryption_enabled = True
        cfg.encryption_provider = "file"
        cfg.encryption_key_path = Path("sqlcipher.key")
    return cfg


def test_sqlite_vector_backend_dense_parity(tmp_path: Path) -> None:
    cfg = _db_config(tmp_path / "vector.db", encrypted=False)
    app_cfg = AppConfig(database=cfg)
    db = DatabaseManager(cfg)
    backend = SqliteVectorBackend(db, dim=3, config=app_cfg)
    upserts = [
        SpanEmbeddingUpsert(
            capture_id="E1",
            span_key="S1",
            vector=[1.0, 0.0, 0.0],
            payload={"app_name": "App", "domain": "example.com"},
            embedding_model="m1",
        ),
        SpanEmbeddingUpsert(
            capture_id="E2",
            span_key="S2",
            vector=[0.0, 1.0, 0.0],
            payload={"app_name": "App", "domain": "example.com"},
            embedding_model="m1",
        ),
        SpanEmbeddingUpsert(
            capture_id="E3",
            span_key="S3",
            vector=[1.0, 1.0, 0.0],
            payload={"app_name": "App", "domain": "example.com"},
            embedding_model="m1",
        ),
        SpanEmbeddingUpsert(
            capture_id="E4",
            span_key="S4",
            vector=[0.0, 0.0, 0.0],
            payload={"app_name": "App", "domain": "example.com"},
            embedding_model="m1",
        ),
    ]
    backend.upsert_spans(upserts)
    query = [1.0, 0.0, 0.0]
    hits = backend.search(query, k=3, embedding_model="m1")
    query_norm = vector_norm(query)
    scored = []
    for item in upserts:
        score = cosine_similarity(
            query,
            item.vector,
            left_norm=query_norm,
            right_norm=vector_norm(item.vector),
        )
        scored.append((item.capture_id, item.span_key, score))
    scored.sort(key=lambda row: (-row[2], row[0], row[1]))
    expected = [(row[0], row[1]) for row in scored[:3]]
    assert [(hit.event_id, hit.span_key) for hit in hits] == expected

    zero_hits = backend.search([0.0, 0.0, 0.0], k=4, embedding_model="m1")
    assert all(abs(hit.score) < 1e-6 for hit in zero_hits)
    assert [hit.event_id for hit in zero_hits] == sorted([u.capture_id for u in upserts])


def test_sqlite_vector_backend_candidate_cap(tmp_path: Path) -> None:
    cfg = _db_config(tmp_path / "vector_cap.db", encrypted=False)
    app_cfg = AppConfig(database=cfg)
    db = DatabaseManager(cfg)
    backend = SqliteVectorBackend(db, dim=2, config=app_cfg)
    upserts = [
        SpanEmbeddingUpsert(
            capture_id=f"E{idx:03d}",
            span_key="S",
            vector=[1.0, 0.0],
            payload={},
            embedding_model="m1",
        )
        for idx in range(250)
    ]
    backend.upsert_spans(upserts)
    _ = backend.search([1.0, 0.0], k=1, embedding_model="m1")
    assert backend.last_candidate_strategy == "signature"
    assert backend.last_candidate_cap == 200
    assert backend.last_candidate_count == 200


def test_sqlcipher_vector_backend_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pysqlcipher3")
    cfg = _db_config(tmp_path / "vector_cipher.db", encrypted=True)
    app_cfg = AppConfig(database=cfg)
    db = DatabaseManager(cfg)
    backend = SqliteVectorBackend(db, dim=2, config=app_cfg)
    upserts = [
        SpanEmbeddingUpsert(
            capture_id="E1",
            span_key="S1",
            vector=[0.2, 0.8],
            payload={},
            embedding_model="m1",
        )
    ]
    backend.upsert_spans(upserts)
    hits = backend.search([0.2, 0.8], k=1, embedding_model="m1")
    assert hits and hits[0].event_id == "E1"
