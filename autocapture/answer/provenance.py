"""Provenance chain helpers for Next-10 answers."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import select

from ..config import AppConfig
from ..storage.database import DatabaseManager
from ..storage.ledger import LedgerWriter
from ..storage.models import (
    ArtifactRecord,
    CitableSpanRecord,
    FrameRecord,
    ProvenanceLedgerEntryRecord,
    RetrievalHitRecord,
)


@dataclass(frozen=True)
class ProvenanceStatus:
    valid_span_ids: set[str]
    missing: dict[str, list[str]]


_TIER_PRIORITY = {"FAST": 1, "FUSION": 2, "RERANK": 3}


def verify_provenance(
    db: DatabaseManager,
    *,
    query_id: str,
    span_ids: list[str],
) -> ProvenanceStatus:
    if not span_ids:
        return ProvenanceStatus(valid_span_ids=set(), missing={})
    span_ids = list(dict.fromkeys(span_ids))
    with db.session() as session:
        spans = (
            session.execute(
                select(CitableSpanRecord).where(CitableSpanRecord.span_id.in_(span_ids))
            )
            .scalars()
            .all()
        )
        spans_by_id = {span.span_id: span for span in spans}
        artifact_ids = [span.artifact_id for span in spans]
        artifacts = (
            session.execute(
                select(ArtifactRecord).where(ArtifactRecord.artifact_id.in_(artifact_ids))
            )
            .scalars()
            .all()
        )
        artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
        frame_ids = [span.frame_id for span in spans]
        frames = (
            session.execute(select(FrameRecord).where(FrameRecord.frame_id.in_(frame_ids)))
            .scalars()
            .all()
        )
        frames_by_id = {frame.frame_id: frame for frame in frames}
        hits = (
            session.execute(
                select(RetrievalHitRecord).where(
                    RetrievalHitRecord.query_id == query_id,
                    RetrievalHitRecord.span_id.in_(span_ids),
                )
            )
            .scalars()
            .all()
        )
        hits_by_span: dict[str, RetrievalHitRecord] = {}
        for hit in hits:
            if not hit.span_id:
                continue
            current = hits_by_span.get(hit.span_id)
            if current is None:
                hits_by_span[hit.span_id] = hit
                continue
            current_rank = _TIER_PRIORITY.get(current.tier, 0)
            hit_rank = _TIER_PRIORITY.get(hit.tier, 0)
            if hit_rank > current_rank or (hit_rank == current_rank and hit.rank < current.rank):
                hits_by_span[hit.span_id] = hit

    valid: set[str] = set()
    missing: dict[str, list[str]] = {}
    for span_id in span_ids:
        reasons: list[str] = []
        span = spans_by_id.get(span_id)
        if span is None or span.tombstoned:
            reasons.append("missing_span")
        else:
            if span.artifact_id not in artifacts_by_id:
                reasons.append("missing_artifact")
            frame = frames_by_id.get(span.frame_id)
            if frame is None or not frame.media_path:
                reasons.append("missing_frame")
        if span_id not in hits_by_span:
            reasons.append("missing_retrieval_hit")
        if reasons:
            missing[span_id] = reasons
        else:
            valid.add(span_id)
    return ProvenanceStatus(valid_span_ids=valid, missing=missing)


def append_provenance_chain(
    config: AppConfig,
    db: DatabaseManager,
    ledger: LedgerWriter,
    *,
    answer_id: str,
    query_id: str,
    evidence_to_span: dict[str, str | None],
    sentence_citations: list[dict],
) -> bool:
    if _answer_chain_exists(db, answer_id):
        return False
    span_ids = [span_id for span_id in evidence_to_span.values() if span_id]
    if not span_ids:
        return False
    span_ids = list(dict.fromkeys(span_ids))
    with db.session() as session:
        spans = (
            session.execute(
                select(CitableSpanRecord).where(CitableSpanRecord.span_id.in_(span_ids))
            )
            .scalars()
            .all()
        )
        spans_by_id = {span.span_id: span for span in spans}
        artifact_ids = [span.artifact_id for span in spans]
        artifacts = (
            session.execute(
                select(ArtifactRecord).where(ArtifactRecord.artifact_id.in_(artifact_ids))
            )
            .scalars()
            .all()
        )
        artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
        frame_ids = [span.frame_id for span in spans]
        frames = (
            session.execute(select(FrameRecord).where(FrameRecord.frame_id.in_(frame_ids)))
            .scalars()
            .all()
        )
        frames_by_id = {frame.frame_id: frame for frame in frames}
        hits = (
            session.execute(
                select(RetrievalHitRecord).where(
                    RetrievalHitRecord.query_id == query_id,
                    RetrievalHitRecord.span_id.in_(span_ids),
                )
            )
            .scalars()
            .all()
        )
        hits_by_span: dict[str, RetrievalHitRecord] = {}
        for hit in hits:
            if not hit.span_id:
                continue
            current = hits_by_span.get(hit.span_id)
            if current is None:
                hits_by_span[hit.span_id] = hit
                continue
            current_rank = _TIER_PRIORITY.get(current.tier, 0)
            hit_rank = _TIER_PRIORITY.get(hit.tier, 0)
            if hit_rank > current_rank or (hit_rank == current_rank and hit.rank < current.rank):
                hits_by_span[hit.span_id] = hit

    created_at = dt.datetime.now(dt.timezone.utc)

    def _next_time(current: dt.datetime) -> dt.datetime:
        return current + dt.timedelta(microseconds=1)

    for span_id in span_ids:
        span = spans_by_id.get(span_id)
        if span is None:
            continue
        artifact = artifacts_by_id.get(span.artifact_id)
        frame = frames_by_id.get(span.frame_id)
        hit = hits_by_span.get(span_id)
        if frame is not None:
            ledger.append_entry(
                "capture",
                {
                    "frame_id": frame.frame_id,
                    "frame_hash": frame.frame_hash,
                    "media_path": frame.media_path,
                },
                answer_id=answer_id,
                created_at=created_at,
            )
            created_at = _next_time(created_at)
        if artifact is not None:
            ledger.append_entry(
                "extract",
                {
                    "artifact_id": artifact.artifact_id,
                    "span_id": span_id,
                    "engine": artifact.engine,
                    "engine_version": artifact.engine_version,
                },
                answer_id=answer_id,
                created_at=created_at,
            )
            created_at = _next_time(created_at)
            ledger.append_entry(
                "index",
                {
                    "index_name": "span_fts",
                    "span_id": span_id,
                    "index_version": config.next10.index_versions.get("span_fts", "v1"),
                },
                answer_id=answer_id,
                created_at=created_at,
            )
            created_at = _next_time(created_at)
        if hit is not None:
            ledger.append_entry(
                "retrieve",
                {
                    "query_id": query_id,
                    "hit_id": hit.hit_id,
                    "tier": hit.tier,
                },
                answer_id=answer_id,
                created_at=created_at,
            )
            created_at = _next_time(created_at)

    for sentence in sentence_citations:
        sentence_id = sentence.get("sentence_id")
        for cite in sentence.get("citations", []):
            span_id = evidence_to_span.get(cite)
            if not span_id:
                continue
            ledger.append_entry(
                "answer_citation",
                {
                    "answer_id": answer_id,
                    "sentence_id": sentence_id,
                    "span_id": span_id,
                    "evidence_id": cite,
                },
                answer_id=answer_id,
                created_at=created_at,
            )
            created_at = _next_time(created_at)
    return True


def _answer_chain_exists(db: DatabaseManager, answer_id: str) -> bool:
    with db.session() as session:
        entry = (
            session.execute(
                select(ProvenanceLedgerEntryRecord.entry_id).where(
                    ProvenanceLedgerEntryRecord.answer_id == answer_id
                )
            )
            .scalars()
            .first()
        )
    return bool(entry)
