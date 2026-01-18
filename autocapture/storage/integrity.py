"""Integrity scan utilities for storage and indexes."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, text

from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import CaptureRecord, EmbeddingRecord, EventRecord, OCRSpanRecord
from ..indexing.vector_index import VectorIndex


@dataclass(frozen=True)
class IntegrityReport:
    orphan_fts: int = 0
    orphan_spans: int = 0
    orphan_embeddings: int = 0
    orphan_vectors: int = 0


def scan_integrity(
    db: DatabaseManager,
    *,
    vector_index: VectorIndex | None = None,
) -> IntegrityReport:
    log = get_logger("integrity.scan")
    with db.session() as session:
        event_ids = {row[0] for row in session.execute(select(EventRecord.event_id)).all()}
        capture_ids = {row[0] for row in session.execute(select(CaptureRecord.id)).all()}

        orphan_spans = session.execute(
            select(OCRSpanRecord.capture_id)
            .where(~OCRSpanRecord.capture_id.in_(capture_ids))
            .distinct()
        ).all()
        orphan_embeddings = session.execute(
            select(EmbeddingRecord.capture_id)
            .where(~EmbeddingRecord.capture_id.in_(capture_ids))
            .distinct()
        ).all()

    orphan_fts = 0
    fts_orphans: list[str] = []
    engine = db.engine
    if engine.dialect.name == "sqlite":
        try:
            with engine.begin() as conn:
                rows = conn.execute(text("SELECT event_id FROM event_fts")).fetchall()
            fts_orphans = [row[0] for row in rows if row[0] not in event_ids]
            orphan_fts = len(fts_orphans)
        except Exception:
            orphan_fts = 0

    orphan_vectors = 0
    if vector_index is not None:
        vector_ids = _list_vector_event_ids(vector_index)
        if vector_ids:
            orphan_vectors = sum(1 for event_id in vector_ids if event_id not in event_ids)

    log.info(
        "Integrity scan complete (fts={}, spans={}, embeddings={}, vectors={})",
        orphan_fts,
        len(orphan_spans),
        len(orphan_embeddings),
        orphan_vectors,
    )
    return IntegrityReport(
        orphan_fts=orphan_fts,
        orphan_spans=len(orphan_spans),
        orphan_embeddings=len(orphan_embeddings),
        orphan_vectors=orphan_vectors,
    )


def find_fts_orphans(db: DatabaseManager) -> list[str]:
    engine = db.engine
    if engine.dialect.name != "sqlite":
        return []
    with db.session() as session:
        event_ids = {row[0] for row in session.execute(select(EventRecord.event_id)).all()}
    try:
        with engine.begin() as conn:
            rows = conn.execute(text("SELECT event_id FROM event_fts")).fetchall()
        return [row[0] for row in rows if row[0] not in event_ids]
    except Exception:
        return []


def _list_vector_event_ids(vector_index: VectorIndex) -> list[str]:
    listing = getattr(vector_index, "list_event_ids", None)
    if callable(listing):
        try:
            return list(listing())
        except Exception:
            return []
    return []
