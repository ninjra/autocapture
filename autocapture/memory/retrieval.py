"""Local retrieval utilities for events and evidence."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
import math
from typing import Iterable

from sqlalchemy import select

from ..config import AppConfig
from ..embeddings.service import EmbeddingService
from ..indexing.lexical_index import LexicalIndex
from ..indexing.vector_index import VectorIndex
from ..logging_utils import get_logger
from ..observability.metrics import retrieval_latency_ms
from ..storage.database import DatabaseManager
from ..storage.models import EventRecord


@dataclass(frozen=True)
class RetrieveFilters:
    apps: list[str] | None = None
    domains: list[str] | None = None


@dataclass(frozen=True)
class RetrievedEvent:
    event: EventRecord
    score: float
    matched_span_keys: list[str] = field(default_factory=list)
    lexical_score: float = 0.0
    vector_score: float = 0.0


class RetrievalService:
    def __init__(
        self,
        db: DatabaseManager,
        config: AppConfig | None = None,
        *,
        embedder: EmbeddingService | None = None,
        vector_index: VectorIndex | None = None,
    ) -> None:
        self._db = db
        self._config = config or AppConfig()
        self._log = get_logger("retrieval")
        self._lexical = LexicalIndex(db)
        self._embedder = embedder or EmbeddingService(self._config.embeddings)
        self._vector = vector_index or VectorIndex(self._config, db, self._embedder.dim)

    def retrieve(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: RetrieveFilters | None,
        limit: int = 12,
    ) -> list[RetrievedEvent]:
        if not query.strip():
            return []
        start = dt.datetime.now(dt.timezone.utc)

        lexical_hits = self._lexical.search(query, limit=limit * 3)
        vector = self._embedder.embed_texts([query])[0]
        vector_hits = self._vector.search(vector, limit=limit * 3)

        lexical_scores = {hit.event_id: hit.score for hit in lexical_hits}
        vector_scores: dict[str, float] = {}
        span_hits: dict[str, list[str]] = {}
        for hit in vector_hits:
            vector_scores[hit.event_id] = max(
                vector_scores.get(hit.event_id, 0.0), hit.score
            )
            span_hits.setdefault(hit.event_id, []).append(hit.span_key)

        candidate_ids = set(lexical_scores) | set(vector_scores)
        if not candidate_ids:
            with self._db.session() as session:
                stmt = select(EventRecord).where(
                    EventRecord.ocr_text.ilike(f"%{query}%")
                )
                if time_range:
                    stmt = stmt.where(EventRecord.ts_start.between(*time_range))
                if filters and filters.apps:
                    stmt = stmt.where(EventRecord.app_name.in_(filters.apps))
                if filters and filters.domains:
                    stmt = stmt.where(EventRecord.domain.in_(filters.domains))
                stmt = stmt.order_by(EventRecord.ts_start.desc()).limit(limit)
                rows = session.execute(stmt).scalars().all()
            return [RetrievedEvent(event=row, score=0.5) for row in rows]

        with self._db.session() as session:
            stmt = select(EventRecord).where(EventRecord.event_id.in_(candidate_ids))
            if time_range:
                stmt = stmt.where(EventRecord.ts_start.between(*time_range))
            if filters and filters.apps:
                stmt = stmt.where(EventRecord.app_name.in_(filters.apps))
            if filters and filters.domains:
                stmt = stmt.where(EventRecord.domain.in_(filters.domains))
            events = session.execute(stmt).scalars().all()

        lexical_norm = _normalize_scores(lexical_scores)
        vector_norm = _normalize_scores(vector_scores)
        now = dt.datetime.now(dt.timezone.utc)

        results: list[RetrievedEvent] = []
        for event in events:
            lex = lexical_norm.get(event.event_id, 0.0)
            vec = vector_norm.get(event.event_id, 0.0)
            event_ts = _ensure_aware(event.ts_start)
            age_hours = max((now - event_ts).total_seconds() / 3600, 0.0)
            recency = _recency_bias(age_hours)
            score = 0.5 * lex + 0.4 * vec + 0.1 * recency
            results.append(
                RetrievedEvent(
                    event=event,
                    score=score,
                    matched_span_keys=span_hits.get(event.event_id, []),
                    lexical_score=lex,
                    vector_score=vec,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        results = results[:limit]
        latency = (dt.datetime.now(dt.timezone.utc) - start).total_seconds() * 1000
        retrieval_latency_ms.observe(latency)
        return results

    def list_events(self, limit: int = 100) -> Iterable[EventRecord]:
        with self._db.session() as session:
            stmt = (
                select(EventRecord).order_by(EventRecord.ts_start.desc()).limit(limit)
            )
            return list(session.execute(stmt).scalars().all())


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {key: 1.0 for key in scores}
    return {
        key: (val - min_score) / (max_score - min_score) for key, val in scores.items()
    }


def _recency_bias(age_hours: float) -> float:
    decay_hours = 72.0
    return float(math.exp(-age_hours / decay_hours))


def _ensure_aware(timestamp: dt.datetime) -> dt.datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp
