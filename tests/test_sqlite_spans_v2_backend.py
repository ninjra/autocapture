from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.indexing.spans_v2 import SparseEmbedding, SpanV2Upsert
from autocapture.indexing.sqlite_backends import SqliteSpansV2Backend
from autocapture.storage.database import DatabaseManager


def _db_config(path: Path, *, encrypted: bool) -> DatabaseConfig:
    cfg = DatabaseConfig(url=f"sqlite:///{path}", sqlite_wal=False)
    if encrypted:
        cfg.encryption_enabled = True
        cfg.encryption_provider = "file"
        cfg.encryption_key_path = Path("sqlcipher.key")
    return cfg


def _payload(app: str = "App") -> dict:
    return {
        "app": app,
        "domain": "example.com",
        "window_title": "Win",
        "frame_id": "F1",
        "frame_hash": "hash",
        "ts": "2026-01-01T00:00:00Z",
        "bbox_norm": [0.1, 0.1, 0.2, 0.2],
        "text": "sample",
        "tags": {"k": "v"},
    }


def test_sqlite_spans_v2_sparse_dot_product(tmp_path: Path) -> None:
    cfg = _db_config(tmp_path / "spans_sparse.db", encrypted=False)
    app_cfg = AppConfig(database=cfg)
    app_cfg.embed.text_model = "m1"
    db = DatabaseManager(cfg)
    backend = SqliteSpansV2Backend(db, dim=2, config=app_cfg)
    upserts = [
        SpanV2Upsert(
            capture_id="E1",
            span_key="S1",
            dense_vector=[1.0, 0.0],
            sparse_vector=SparseEmbedding(indices=[1, 2], values=[0.2, 0.2]),
            late_vectors=None,
            payload=_payload(),
            embedding_model="m1",
        ),
        SpanV2Upsert(
            capture_id="E2",
            span_key="S2",
            dense_vector=[0.0, 1.0],
            sparse_vector=SparseEmbedding(indices=[1, 3], values=[1.0, 0.1]),
            late_vectors=None,
            payload=_payload(),
            embedding_model="m1",
        ),
    ]
    backend.upsert(upserts)
    query = SparseEmbedding(indices=[1, 2], values=[1.0, 1.0])
    hits = backend.search_sparse(query, k=2, filters={"app": ["App"]})
    assert [hit.event_id for hit in hits] == ["E2", "E1"]


def test_sqlite_spans_v2_late_maxsim_ordering(tmp_path: Path) -> None:
    cfg = _db_config(tmp_path / "spans_late.db", encrypted=False)
    app_cfg = AppConfig(database=cfg)
    app_cfg.embed.text_model = "m1"
    db = DatabaseManager(cfg)
    backend = SqliteSpansV2Backend(db, dim=2, config=app_cfg)
    upserts = [
        SpanV2Upsert(
            capture_id="E1",
            span_key="S1",
            dense_vector=[1.0, 0.0],
            sparse_vector=None,
            late_vectors=[[1.0, 0.0], [0.0, 1.0]],
            payload=_payload(),
            embedding_model="m1",
        ),
        SpanV2Upsert(
            capture_id="E2",
            span_key="S2",
            dense_vector=[0.0, 1.0],
            sparse_vector=None,
            late_vectors=[[0.5, 0.0], [0.0, 0.5]],
            payload=_payload(),
            embedding_model="m1",
        ),
    ]
    backend.upsert(upserts)
    hits = backend.search_late([[1.0, 0.0], [0.0, 1.0]], k=2, filters={"app": ["App"]})
    assert [hit.event_id for hit in hits] == ["E1", "E2"]
    assert backend.last_candidate_count <= backend.last_candidate_cap


def test_sqlite_spans_v2_delete_cascade(tmp_path: Path) -> None:
    cfg = _db_config(tmp_path / "spans_delete.db", encrypted=False)
    app_cfg = AppConfig(database=cfg)
    app_cfg.embed.text_model = "m1"
    db = DatabaseManager(cfg)
    backend = SqliteSpansV2Backend(db, dim=2, config=app_cfg)
    backend.upsert(
        [
            SpanV2Upsert(
                capture_id="E1",
                span_key="S1",
                dense_vector=[1.0, 0.0],
                sparse_vector=None,
                late_vectors=[[1.0, 0.0]],
                payload=_payload(),
                embedding_model="m1",
            )
        ]
    )
    deleted = backend.delete_event_ids(["E1"])
    assert deleted == 1
    with db.engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM vec_spans_v2_late")).scalar()
    assert count == 0


def test_sqlcipher_spans_v2_backend_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pysqlcipher3")
    cfg = _db_config(tmp_path / "spans_cipher.db", encrypted=True)
    app_cfg = AppConfig(database=cfg)
    app_cfg.embed.text_model = "m1"
    db = DatabaseManager(cfg)
    backend = SqliteSpansV2Backend(db, dim=2, config=app_cfg)
    backend.upsert(
        [
            SpanV2Upsert(
                capture_id="E1",
                span_key="S1",
                dense_vector=[0.1, 0.9],
                sparse_vector=None,
                late_vectors=[[0.1, 0.9]],
                payload=_payload(),
                embedding_model="m1",
            )
        ]
    )
    hits = backend.search_dense([0.1, 0.9], k=1, filters=None, embedding_model="m1")
    assert hits and hits[0].event_id == "E1"
