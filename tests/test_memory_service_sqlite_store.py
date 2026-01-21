from __future__ import annotations

import datetime as dt
import threading
from pathlib import Path

import pytest
from sqlalchemy import text

from autocapture.config import AppConfig, DatabaseConfig, MemoryServiceConfig
from autocapture.memory_service.app import resolve_memory_service_db_url
from autocapture.memory_service.providers import HashEmbedder
from autocapture.memory_service.schemas import (
    MemoryIngestRequest,
    MemoryPolicyContext,
    MemoryProposal,
    PolicyLabels,
    ProvenancePointer,
    MemoryQueryRequest,
)
from autocapture.memory_service.store import SqliteMemoryServiceStore
from autocapture.memory_service.utils import hash_text
from autocapture.storage.database import DatabaseManager


def _db_config(path: Path, *, encrypted: bool) -> DatabaseConfig:
    cfg = DatabaseConfig(url=f"sqlite:///{path}", sqlite_wal=False)
    if encrypted:
        cfg.encryption_enabled = True
        cfg.encryption_provider = "file"
        cfg.encryption_key_path = Path("sqlcipher.key")
    return cfg


def _seed_provenance(db: DatabaseManager, *, namespace: str) -> ProvenancePointer:
    artifact_version_id = "artifact_v1"
    chunk_id = "chunk_1"
    excerpt = "hello world"
    excerpt_hash = hash_text(excerpt)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    with db.session() as session:
        session.execute(
            text(
                "INSERT INTO artifact_versions(artifact_version_id, namespace, artifact_id, "
                "content_hash, source_uri, title, labels_json, metadata_json, created_at) "
                "VALUES (:artifact_version_id, :namespace, :artifact_id, :content_hash, :source_uri, "
                ":title, :labels_json, :metadata_json, :created_at)"
            ),
            {
                "artifact_version_id": artifact_version_id,
                "namespace": namespace,
                "artifact_id": "artifact_1",
                "content_hash": hash_text("artifact payload"),
                "source_uri": "local://test",
                "title": "title",
                "labels_json": "[]",
                "metadata_json": "{}",
                "created_at": now,
            },
        )
        session.execute(
            text(
                "INSERT INTO artifact_chunks(chunk_id, artifact_version_id, namespace, start_offset, "
                "end_offset, excerpt_hash, created_at) VALUES ("
                ":chunk_id, :artifact_version_id, :namespace, :start_offset, :end_offset, "
                ":excerpt_hash, :created_at)"
            ),
            {
                "chunk_id": chunk_id,
                "artifact_version_id": artifact_version_id,
                "namespace": namespace,
                "start_offset": 0,
                "end_offset": 5,
                "excerpt_hash": excerpt_hash,
                "created_at": now,
            },
        )
    return ProvenancePointer(
        artifact_version_id=artifact_version_id,
        chunk_id=chunk_id,
        start_offset=0,
        end_offset=5,
        excerpt_hash=excerpt_hash,
    )


def _build_store(
    tmp_path: Path, *, encrypted: bool
) -> tuple[SqliteMemoryServiceStore, DatabaseManager]:
    cfg = _db_config(tmp_path / "memory.db", encrypted=encrypted)
    mem_cfg = MemoryServiceConfig()
    db = DatabaseManager(cfg)
    store = SqliteMemoryServiceStore(db, mem_cfg, HashEmbedder(dim=mem_cfg.embedder.dim), None)
    return store, db


def test_memory_service_sqlite_ingest_query_roundtrip(tmp_path: Path) -> None:
    store, db = _build_store(tmp_path, encrypted=False)
    namespace = "default"
    pointer = _seed_provenance(db, namespace=namespace)
    proposal = MemoryProposal(
        memory_type="fact",
        content_text="hello world",
        policy=PolicyLabels(audience=["internal"], sensitivity="low"),
        provenance=[pointer],
    )
    ingest = store.ingest(MemoryIngestRequest(namespace=namespace, proposals=[proposal]))
    assert ingest.accepted == 1
    request = MemoryQueryRequest(
        namespace=namespace,
        query="hello",
        policy=MemoryPolicyContext(audience=["internal"], sensitivity_max="high"),
    )
    response = store.query(request)
    assert response.cards


def test_memory_service_sqlite_health_reports_tables(tmp_path: Path) -> None:
    store, _db = _build_store(tmp_path, encrypted=False)
    health = store.health()
    assert health["db_connected"] is True
    assert health["tables_ready"] is True
    assert health["pgvector"] is False


def test_memory_service_db_paths_are_distinct_by_default(tmp_path: Path) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'main.db'}"))
    mem_url = resolve_memory_service_db_url(config)
    assert mem_url != config.database.url


def test_memory_service_sqlite_concurrent_ingest_query(tmp_path: Path) -> None:
    store, db = _build_store(tmp_path, encrypted=False)
    namespace = "default"
    pointer = _seed_provenance(db, namespace=namespace)
    proposal = MemoryProposal(
        memory_type="fact",
        content_text="threaded memory",
        policy=PolicyLabels(audience=["internal"], sensitivity="low"),
        provenance=[pointer],
    )

    def _ingest():
        store.ingest(MemoryIngestRequest(namespace=namespace, proposals=[proposal]))

    def _query():
        request = MemoryQueryRequest(
            namespace=namespace,
            query="threaded",
            policy=MemoryPolicyContext(audience=["internal"], sensitivity_max="high"),
        )
        store.query(request)

    threads = [threading.Thread(target=_ingest) for _ in range(2)] + [
        threading.Thread(target=_query) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_memory_service_sqlcipher_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pysqlcipher3")
    store, db = _build_store(tmp_path, encrypted=True)
    namespace = "default"
    pointer = _seed_provenance(db, namespace=namespace)
    proposal = MemoryProposal(
        memory_type="fact",
        content_text="encrypted memory",
        policy=PolicyLabels(audience=["internal"], sensitivity="low"),
        provenance=[pointer],
    )
    ingest = store.ingest(MemoryIngestRequest(namespace=namespace, proposals=[proposal]))
    assert ingest.accepted == 1
    request = MemoryQueryRequest(
        namespace=namespace,
        query="encrypted",
        policy=MemoryPolicyContext(audience=["internal"], sensitivity_max="high"),
    )
    response = store.query(request)
    assert response.cards
