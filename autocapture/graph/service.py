"""Graph retrieval service implementation."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Iterable

from sqlalchemy import select

from ..config import AppConfig
from ..logging_utils import get_logger
from ..memory.retrieval import RetrieveFilters, RetrievalService
from ..memory.threads import ThreadCandidate, ThreadRetrievalService, ThreadSegmenter, ThreadStore
from ..storage.database import DatabaseManager
from ..storage.models import EventRecord, ThreadEventRecord
from .models import (
    GraphFilters,
    GraphHit,
    GraphIndexRequest,
    GraphIndexResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    GraphTimeRange,
)
from .workers import GraphWorkerGroup, GraphWorkerSpec

_CORPUS_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class GraphService:
    def __init__(
        self,
        config: AppConfig,
        *,
        db: DatabaseManager | None = None,
        thread_retrieval: ThreadRetrievalService | None = None,
        retrieval: RetrievalService | None = None,
        segmenter: ThreadSegmenter | None = None,
        thread_store: ThreadStore | None = None,
    ) -> None:
        self._config = config
        self._db = db or DatabaseManager(config.database)
        self._log = get_logger("graph.service")
        self._thread_retrieval = thread_retrieval
        self._retrieval = retrieval
        self._segmenter = segmenter or ThreadSegmenter(
            max_gap_minutes=config.threads.max_gap_minutes,
            app_similarity=config.threads.app_similarity_threshold,
            title_similarity=config.threads.title_similarity_threshold,
        )
        self._thread_store = thread_store or ThreadStore(self._db)
        self._workspace_root = Path(config.graph_service.workspace_root)
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        specs = [
            GraphWorkerSpec(
                name="graphrag",
                cli_path=config.graph_service.graphrag_cli,
                timeout_s=config.graph_service.worker_timeout_s,
            ),
            GraphWorkerSpec(
                name="hypergraphrag",
                cli_path=config.graph_service.hypergraphrag_cli,
                timeout_s=config.graph_service.worker_timeout_s,
            ),
            GraphWorkerSpec(
                name="hyperrag",
                cli_path=config.graph_service.hyperrag_cli,
                timeout_s=config.graph_service.worker_timeout_s,
            ),
        ]
        self._workers = GraphWorkerGroup(specs, workspace_root=self._workspace_root)

    def index(
        self, request: GraphIndexRequest, *, adapter: str | None = None, use_workers: bool = True
    ) -> GraphIndexResponse:
        adapter = (adapter or "graphrag").strip().lower()
        if use_workers and self._config.graph_service.require_workers:
            return self._workers.index(adapter, request)
        corpus_id = _safe_corpus_id(request.corpus_id)
        time_range = _parse_time_range(request.time_range)
        if time_range is None:
            now = dt.datetime.now(dt.timezone.utc)
            window_days = int(self._config.retention.screenshot_ttl_days)
            time_range = (now - dt.timedelta(days=window_days), now)
        max_events = int(
            min(
                request.max_events or self._config.graph_service.max_events,
                self._config.graph_service.max_events,
            )
        )
        events = self._load_events(time_range, request.filters, max_events)
        segments = self._segmenter.segment(events)
        if segments:
            self._thread_store.persist(segments)
        self._write_manifest(corpus_id, time_range, events, segments)
        return GraphIndexResponse(
            status="ok",
            corpus_id=corpus_id,
            events_indexed=len(events),
            segments=len(segments),
        )

    def query(
        self, request: GraphQueryRequest, *, adapter: str | None = None, use_workers: bool = True
    ) -> GraphQueryResponse:
        adapter = (adapter or "graphrag").strip().lower()
        if use_workers and self._config.graph_service.require_workers:
            return self._workers.query(adapter, request)
        corpus_id = _safe_corpus_id(request.corpus_id)
        _ = corpus_id
        time_range = _parse_time_range(request.time_range)
        limit = min(int(request.limit), int(self._config.graph_service.max_results))
        thread_limit = max(limit, 5)
        try:
            if self._thread_retrieval is None:
                self._thread_retrieval = ThreadRetrievalService(self._config, self._db)
            threads = self._thread_retrieval.retrieve(request.query, time_range, limit=thread_limit)
        except Exception as exc:
            self._log.warning("Thread retrieval failed: {}", exc)
            threads = []
        hits = self._thread_hits(threads, time_range, request.filters, limit)
        if not hits:
            hits = self._fallback_hits(request, time_range, limit)
        return GraphQueryResponse(hits=hits)

    def _load_events(
        self,
        time_range: tuple[dt.datetime, dt.datetime],
        filters: GraphFilters | None,
        limit: int,
    ) -> list[EventRecord]:
        with self._db.session() as session:
            stmt = select(EventRecord)
            if time_range:
                stmt = stmt.where(EventRecord.ts_start.between(*time_range))
            if filters and filters.apps:
                stmt = stmt.where(EventRecord.app_name.in_(filters.apps))
            if filters and filters.domains:
                stmt = stmt.where(EventRecord.domain.in_(filters.domains))
            stmt = stmt.order_by(EventRecord.ts_start.desc()).limit(limit)
            return list(session.execute(stmt).scalars().all())

    def _thread_hits(
        self,
        threads: Iterable[ThreadCandidate],
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: GraphFilters | None,
        limit: int,
    ) -> list[GraphHit]:
        candidates = list(threads)
        if not candidates:
            return []
        thread_ids = [candidate.thread_id for candidate in candidates]
        thread_events = self._load_thread_events(thread_ids)
        if not thread_events:
            return []
        all_event_ids = {event_id for ids in thread_events.values() for event_id in ids}
        allowed_event_ids = self._filter_event_ids(all_event_ids, time_range, filters)
        scores: dict[str, float] = {}
        snippets: dict[str, str | None] = {}
        for candidate in candidates:
            event_ids = thread_events.get(candidate.thread_id, [])
            for position, event_id in enumerate(event_ids):
                if allowed_event_ids and event_id not in allowed_event_ids:
                    continue
                score = max(candidate.score - (position * 0.01), 0.0)
                if score > scores.get(event_id, -1.0):
                    scores[event_id] = score
                    snippets[event_id] = candidate.summary or candidate.title
        hits = [
            GraphHit(event_id=event_id, score=score, snippet=snippets.get(event_id))
            for event_id, score in scores.items()
        ]
        hits.sort(key=lambda item: (-item.score, item.event_id))
        return hits[:limit]

    def _fallback_hits(
        self,
        request: GraphQueryRequest,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        limit: int,
    ) -> list[GraphHit]:
        if self._retrieval is None:
            self._retrieval = RetrievalService(self._db, self._config)
        filters = None
        if request.filters:
            filters = RetrieveFilters(
                apps=request.filters.apps,
                domains=request.filters.domains,
            )
        batch = self._retrieval.retrieve(
            request.query,
            time_range,
            filters,
            limit=limit,
        )
        hits: list[GraphHit] = []
        for item in batch.results:
            hits.append(
                GraphHit(
                    event_id=item.event.event_id,
                    score=item.score,
                    snippet=item.snippet,
                )
            )
        return hits

    def _filter_event_ids(
        self,
        event_ids: set[str],
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: GraphFilters | None,
    ) -> set[str]:
        if not event_ids:
            return set()
        if not time_range and not (filters and (filters.apps or filters.domains)):
            return set(event_ids)
        with self._db.session() as session:
            stmt = select(EventRecord.event_id).where(EventRecord.event_id.in_(event_ids))
            if time_range:
                stmt = stmt.where(EventRecord.ts_start.between(*time_range))
            if filters and filters.apps:
                stmt = stmt.where(EventRecord.app_name.in_(filters.apps))
            if filters and filters.domains:
                stmt = stmt.where(EventRecord.domain.in_(filters.domains))
            rows = session.execute(stmt).scalars().all()
        return {str(row) for row in rows}

    def _load_thread_events(self, thread_ids: list[str]) -> dict[str, list[str]]:
        if not thread_ids:
            return {}
        with self._db.session() as session:
            rows = (
                session.execute(
                    select(ThreadEventRecord)
                    .where(ThreadEventRecord.thread_id.in_(thread_ids))
                    .order_by(ThreadEventRecord.thread_id.asc(), ThreadEventRecord.position.asc())
                )
                .scalars()
                .all()
            )
        mapping: dict[str, list[str]] = {}
        for row in rows:
            mapping.setdefault(row.thread_id, []).append(row.event_id)
        return mapping

    def _write_manifest(
        self,
        corpus_id: str,
        time_range: tuple[dt.datetime, dt.datetime],
        events: list[EventRecord],
        segments: list,
    ) -> None:
        workspace = self._workspace_root / corpus_id
        workspace.mkdir(parents=True, exist_ok=True)
        manifest = {
            "corpus_id": corpus_id,
            "indexed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "events_indexed": len(events),
            "segments": len(segments),
            "time_range": {
                "start": time_range[0].isoformat() if time_range else None,
                "end": time_range[1].isoformat() if time_range else None,
            },
        }
        path = workspace / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_corpus_id(value: str) -> str:
    cleaned = _CORPUS_SAFE.sub("_", (value or "").strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        return "default"
    return cleaned[:64]


def _parse_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _parse_time_range(
    time_range: GraphTimeRange | None,
) -> tuple[dt.datetime, dt.datetime] | None:
    if time_range is None:
        return None
    start = _parse_datetime(time_range.start)
    end = _parse_datetime(time_range.end)
    if start is None or end is None:
        raise ValueError("Invalid time_range; expected ISO8601 start/end")
    if start > end:
        start, end = end, start
    return start, end


__all__ = ["GraphService"]
