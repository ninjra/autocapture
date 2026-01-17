"""Thread segmentation and retrieval utilities."""

from __future__ import annotations

import datetime as dt
import re
import uuid
from dataclasses import dataclass, field
from typing import Iterable

from sqlalchemy import delete, select

from ..config import AppConfig
from ..embeddings.service import EmbeddingService
from ..indexing.thread_index import ThreadLexicalIndex
from ..indexing.vector_index import VectorIndex, VectorHit
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import EventRecord, ThreadEventRecord, ThreadRecord, ThreadSummaryRecord

THREAD_NAMESPACE = uuid.UUID("2f5c1c62-6e3d-4f2b-95c8-0b0f8d8bbd1c")


@dataclass(frozen=True)
class ThreadSegment:
    thread_id: str
    event_ids: list[str]
    ts_start: dt.datetime
    ts_end: dt.datetime
    app_name: str
    window_title: str


@dataclass(frozen=True)
class ThreadCandidate:
    thread_id: str
    score: float
    lexical_score: float
    vector_score: float
    title: str
    summary: str
    ts_start: dt.datetime
    ts_end: dt.datetime | None
    citations: list[dict] = field(default_factory=list)


class ThreadSegmenter:
    def __init__(
        self, *, max_gap_minutes: float, app_similarity: float, title_similarity: float
    ) -> None:
        self._max_gap_minutes = max_gap_minutes
        self._app_similarity = app_similarity
        self._title_similarity = title_similarity

    def segment(self, events: Iterable[EventRecord]) -> list[ThreadSegment]:
        ordered = sorted(events, key=lambda event: event.ts_start)
        segments: list[ThreadSegment] = []
        current_events: list[EventRecord] = []
        for event in ordered:
            if not current_events:
                current_events = [event]
                continue
            if self._should_split(current_events[-1], event):
                segments.append(_build_segment(current_events))
                current_events = [event]
            else:
                current_events.append(event)
        if current_events:
            segments.append(_build_segment(current_events))
        return segments

    def _should_split(self, last_event: EventRecord, next_event: EventRecord) -> bool:
        gap_minutes = (
            (next_event.ts_start - last_event.ts_start).total_seconds() / 60.0
            if next_event.ts_start and last_event.ts_start
            else 0.0
        )
        if gap_minutes > self._max_gap_minutes:
            return True
        app_sim = _token_similarity(last_event.app_name, next_event.app_name)
        title_sim = _token_similarity(last_event.window_title, next_event.window_title)
        if app_sim < self._app_similarity and title_sim < self._title_similarity:
            return True
        return False


class ThreadStore:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._log = get_logger("threads.store")

    def persist(self, segments: list[ThreadSegment]) -> None:
        if not segments:
            return
        event_ids = [event_id for segment in segments for event_id in segment.event_ids]

        def _write(session) -> None:
            session.execute(
                delete(ThreadEventRecord).where(ThreadEventRecord.event_id.in_(event_ids))
            )
            now = dt.datetime.now(dt.timezone.utc)
            for segment in segments:
                record = session.get(ThreadRecord, segment.thread_id)
                if record:
                    record.ts_start = segment.ts_start
                    record.ts_end = segment.ts_end
                    record.app_name = segment.app_name
                    record.window_title = segment.window_title
                    record.event_count = len(segment.event_ids)
                    record.updated_at = now
                else:
                    session.add(
                        ThreadRecord(
                            thread_id=segment.thread_id,
                            ts_start=segment.ts_start,
                            ts_end=segment.ts_end,
                            app_name=segment.app_name,
                            window_title=segment.window_title,
                            event_count=len(segment.event_ids),
                            created_at=now,
                            updated_at=now,
                        )
                    )
            session.flush()
            for segment in segments:
                for position, event_id in enumerate(segment.event_ids):
                    session.add(
                        ThreadEventRecord(
                            thread_id=segment.thread_id,
                            event_id=event_id,
                            position=position,
                        )
                    )

        self._db.transaction(_write)


class ThreadRetrievalService:
    def __init__(
        self,
        config: AppConfig,
        db: DatabaseManager,
        *,
        embedder: EmbeddingService | None = None,
        vector_index: VectorIndex | None = None,
        lexical_index: ThreadLexicalIndex | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._embedder = embedder or EmbeddingService(config.embed)
        self._vector_index = vector_index or VectorIndex(config, self._embedder.dim)
        self._lexical = lexical_index or ThreadLexicalIndex(db)

    def retrieve(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        limit: int = 5,
    ) -> list[ThreadCandidate]:
        query = query.strip()
        if not query:
            return self._recent_threads(time_range, limit)
        limit = max(1, limit)
        candidate_limit = limit * 3
        lexical_hits = self._lexical.search(query, limit=candidate_limit)
        vector_hits: list[VectorHit] = []
        try:
            vector = self._embedder.embed_texts([query])[0]
            vector_hits = self._vector_index.search(
                vector,
                candidate_limit,
                filters={"kind": "thread"},
                embedding_model=self._embedder.model_name,
            )
        except Exception:
            vector_hits = []
        lexical_scores = {hit.thread_id: hit.score for hit in lexical_hits}
        vector_scores = {hit.event_id: hit.score for hit in vector_hits}
        candidate_ids = set(lexical_scores) | set(vector_scores)
        if not candidate_ids:
            return []

        with self._db.session() as session:
            stmt = select(ThreadRecord).where(ThreadRecord.thread_id.in_(candidate_ids))
            if time_range:
                stmt = stmt.where(ThreadRecord.ts_start.between(*time_range))
            threads = session.execute(stmt).scalars().all()
            summaries = (
                session.execute(
                    select(ThreadSummaryRecord).where(
                        ThreadSummaryRecord.thread_id.in_([t.thread_id for t in threads])
                    )
                )
                .scalars()
                .all()
            )
        summary_map = {summary.thread_id: summary for summary in summaries}
        results: list[ThreadCandidate] = []
        for thread in threads:
            lex = lexical_scores.get(thread.thread_id, 0.0)
            vec = vector_scores.get(thread.thread_id, 0.0)
            score = 0.6 * lex + 0.4 * vec
            summary_record = summary_map.get(thread.thread_id)
            payload = summary_record.data_json if summary_record else {}
            title = str(payload.get("title") or thread.window_title or "")
            summary_text = str(payload.get("summary") or "")
            citations = payload.get("citations") or []
            results.append(
                ThreadCandidate(
                    thread_id=thread.thread_id,
                    score=score,
                    lexical_score=lex,
                    vector_score=vec,
                    title=title,
                    summary=summary_text,
                    ts_start=thread.ts_start,
                    ts_end=thread.ts_end,
                    citations=citations if isinstance(citations, list) else [],
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def _recent_threads(
        self, time_range: tuple[dt.datetime, dt.datetime] | None, limit: int
    ) -> list[ThreadCandidate]:
        with self._db.session() as session:
            stmt = select(ThreadRecord)
            if time_range:
                stmt = stmt.where(ThreadRecord.ts_start.between(*time_range))
            stmt = stmt.order_by(ThreadRecord.ts_start.desc()).limit(limit)
            threads = session.execute(stmt).scalars().all()
            summaries = (
                session.execute(
                    select(ThreadSummaryRecord).where(
                        ThreadSummaryRecord.thread_id.in_([t.thread_id for t in threads])
                    )
                )
                .scalars()
                .all()
            )
        summary_map = {summary.thread_id: summary for summary in summaries}
        results: list[ThreadCandidate] = []
        for thread in threads:
            summary_record = summary_map.get(thread.thread_id)
            payload = summary_record.data_json if summary_record else {}
            results.append(
                ThreadCandidate(
                    thread_id=thread.thread_id,
                    score=0.4,
                    lexical_score=0.0,
                    vector_score=0.0,
                    title=str(payload.get("title") or thread.window_title or ""),
                    summary=str(payload.get("summary") or ""),
                    ts_start=thread.ts_start,
                    ts_end=thread.ts_end,
                    citations=(
                        payload.get("citations")
                        if isinstance(payload.get("citations"), list)
                        else []
                    ),
                )
            )
        return results


def _build_segment(events: list[EventRecord]) -> ThreadSegment:
    first = events[0]
    last = events[-1]
    thread_id = str(uuid.uuid5(THREAD_NAMESPACE, first.event_id))
    return ThreadSegment(
        thread_id=thread_id,
        event_ids=[event.event_id for event in events],
        ts_start=first.ts_start,
        ts_end=last.ts_start,
        app_name=first.app_name,
        window_title=first.window_title,
    )


def _token_similarity(a: str | None, b: str | None) -> float:
    tokens_a = set(re.findall(r"[A-Za-z0-9_]+", (a or "").lower()))
    tokens_b = set(re.findall(r"[A-Za-z0-9_]+", (b or "").lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)
