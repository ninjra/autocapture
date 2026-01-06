from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from autocapture.config import AppConfig, DatabaseConfig, WorkerConfig
from autocapture.indexing.vector_index import HNSWIndex
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import HNSWMappingRecord


def test_hnsw_mapping_unique_and_atomic_save(tmp_path: Path) -> None:
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
        worker=WorkerConfig(data_dir=tmp_path),
    )
    db = DatabaseManager(config.database)
    index = HNSWIndex(config, db, dim=3)

    payload = {"span_id": 1}
    index.upsert_spans([("event-1", "S1", [0.1, 0.2, 0.3], payload)])
    index.upsert_spans([("event-1", "S1", [0.1, 0.2, 0.3], payload)])

    with db.session() as session:
        rows = (
            session.execute(
                select(HNSWMappingRecord).where(
                    HNSWMappingRecord.event_id == "event-1",
                    HNSWMappingRecord.span_key == "S1",
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1

    index_path = tmp_path / "embeddings" / "hnsw.index"
    temp_path = tmp_path / "embeddings" / "hnsw.tmp"
    assert index_path.exists()
    assert not temp_path.exists()
