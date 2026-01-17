"""Local retrieval utilities for events and evidence."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
import inspect
import math
import time
import re
from typing import Iterable

from sqlalchemy import select

from ..config import AppConfig
from ..embeddings.service import EmbeddingService
from ..embeddings.sparse import SparseEncoder
from ..embeddings.late import LateInteractionEncoder
from ..indexing.lexical_index import LexicalIndex
from ..indexing.vector_index import VectorHit, VectorIndex
from ..indexing.spans_v2 import SpansV2Index
from ..logging_utils import get_logger
from ..observability.metrics import retrieval_latency_ms, vector_search_failures_total
from ..storage.database import DatabaseManager
from ..storage.models import EventRecord, RetrievalTraceRecord
from .reranker import CrossEncoderReranker
from ..runtime_governor import RuntimeGovernor, RuntimeMode


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
    sparse_score: float = 0.0
    late_score: float = 0.0
    engine: str = "hybrid"
    rank: int = 0
    rank_gap: float = 0.0


class RetrievalService:
    def __init__(
        self,
        db: DatabaseManager,
        config: AppConfig | None = None,
        *,
        embedder: EmbeddingService | None = None,
        vector_index: VectorIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
        spans_index: SpansV2Index | None = None,
        runtime_governor: RuntimeGovernor | None = None,
    ) -> None:
        self._db = db
        self._config = config or AppConfig()
        self._log = get_logger("retrieval")
        self._lexical = LexicalIndex(db)
        self._embedder = embedder or EmbeddingService(self._config.embed)
        self._vector = vector_index or VectorIndex(self._config, self._embedder.dim)
        self._spans_v2 = spans_index
        if self._spans_v2 is None and (
            self._config.retrieval.use_spans_v2
            or self._config.retrieval.sparse_enabled
            or self._config.retrieval.late_enabled
        ):
            self._spans_v2 = SpansV2Index(self._config, self._embedder.dim)
        self._sparse_encoder: SparseEncoder | None = None
        self._late_encoder: LateInteractionEncoder | None = None
        if self._config.retrieval.sparse_enabled:
            self._sparse_encoder = SparseEncoder(self._config.retrieval.sparse_model)
        if self._config.retrieval.late_enabled:
            self._late_encoder = LateInteractionEncoder(
                dim=int(self._config.qdrant.late_vector_size), max_tokens=64
            )
        self._reranker = reranker
        self._reranker_failed = False
        self._last_vector_failure_log = 0.0
        self._last_reranker_failure_log = 0.0
        self._runtime = runtime_governor

    def retrieve(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: RetrieveFilters | None,
        limit: int = 12,
        offset: int = 0,
        mode: str | None = None,
    ) -> list[RetrievedEvent]:
        query = query.strip()
        if len(query) < 2:
            if time_range:
                return self._retrieve_time_range(time_range, filters, limit, offset)
            return []
        limit = max(1, limit)
        offset = max(0, offset)
        start = dt.datetime.now(dt.timezone.utc)
        mode_value = (mode or "auto").strip().lower()

        candidate_limit = (limit + offset) * 3
        baseline = self._retrieve_candidates(
            query, time_range, filters, candidate_limit, engine="baseline"
        )
        results = baseline
        rewrites: list[str] = []

        enable_fusion = self._config.retrieval.fusion_enabled and mode_value in {
            "auto",
            "deep",
        }
        if enable_fusion:
            confidence = _retrieval_confidence(baseline)
            if mode_value == "deep" or not _is_confident(
                confidence,
                self._config.retrieval.fusion_confidence_min,
                self._config.retrieval.fusion_rank_gap_min,
            ):
                rewrites = self._rewrite_queries(query)
                fused_lists = [baseline]
                for rewrite in rewrites:
                    if rewrite.strip().lower() == query.strip().lower():
                        continue
                    fused_lists.append(
                        self._retrieve_candidates(
                            rewrite, time_range, filters, candidate_limit, engine="rewrite"
                        )
                    )
                results = _rrf_fuse(fused_lists, self._config.retrieval.fusion_rrf_k)

        if self._config.retrieval.late_enabled and mode_value in {"auto", "deep"}:
            results = self._late_rerank(query, results, candidate_limit)

        results = self._rerank_results(query, results)
        results = _assign_ranks(results)
        results = results[offset : offset + limit]
        latency = (dt.datetime.now(dt.timezone.utc) - start).total_seconds() * 1000
        retrieval_latency_ms.observe(latency)
        if self._config.retrieval.traces_enabled:
            self._persist_trace(query, rewrites, results)
        return results

    def list_events(self, limit: int = 100) -> Iterable[EventRecord]:
        with self._db.session() as session:
            stmt = select(EventRecord).order_by(EventRecord.ts_start.desc()).limit(limit)
            return list(session.execute(stmt).scalars().all())

    def _retrieve_time_range(
        self,
        time_range: tuple[dt.datetime, dt.datetime],
        filters: RetrieveFilters | None,
        limit: int,
        offset: int,
    ) -> list[RetrievedEvent]:
        with self._db.session() as session:
            stmt = select(EventRecord).where(EventRecord.ts_start.between(*time_range))
            if filters and filters.apps:
                stmt = stmt.where(EventRecord.app_name.in_(filters.apps))
            if filters and filters.domains:
                stmt = stmt.where(EventRecord.domain.in_(filters.domains))
            stmt = stmt.order_by(EventRecord.ts_start.desc()).offset(offset).limit(limit)
            rows = session.execute(stmt).scalars().all()
        return [RetrievedEvent(event=row, score=0.4) for row in rows]

    def _retrieve_candidates(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: RetrieveFilters | None,
        limit: int,
        *,
        engine: str,
    ) -> list[RetrievedEvent]:
        lexical_hits = self._lexical.search(query, limit=limit)
        lexical_scores = {hit.event_id: hit.score for hit in lexical_hits}
        dense_hits: list[VectorHit] = []
        sparse_hits: list[VectorHit] = []
        span_hits: dict[str, list[str]] = {}

        dense_vector = None
        try:
            dense_vector = self._embedder.embed_texts([query])[0]
        except Exception as exc:
            vector_search_failures_total.inc()
            now = time.monotonic()
            if now - self._last_vector_failure_log > 5.0:
                self._last_vector_failure_log = now
                self._log.warning("Dense embedding failed; using lexical-only results: {}", exc)

        filters_map = _build_vector_filters(filters, v2=False)
        filters_v2 = _build_vector_filters(filters, v2=True)

        if dense_vector is not None:
            try:
                if self._spans_v2 and self._config.retrieval.use_spans_v2:
                    dense_hits = self._spans_v2.search_dense(
                        dense_vector,
                        limit,
                        filters=filters_v2,
                        embedding_model=self._embedder.model_name,
                    )
                else:
                    dense_hits = self._vector.search(
                        dense_vector,
                        limit,
                        filters=filters_map,
                        embedding_model=self._embedder.model_name,
                    )
            except Exception as exc:
                vector_search_failures_total.inc()
                now = time.monotonic()
                if now - self._last_vector_failure_log > 5.0:
                    self._last_vector_failure_log = now
                    self._log.warning(
                        "Vector retrieval failed; using lexical-only results: {}", exc
                    )

        dense_scores: dict[str, float] = {}
        for hit in dense_hits:
            dense_scores[hit.event_id] = max(dense_scores.get(hit.event_id, 0.0), hit.score)
            span_hits.setdefault(hit.event_id, []).append(hit.span_key)

        sparse_scores: dict[str, float] = {}
        if self._config.retrieval.sparse_enabled and self._spans_v2 and self._sparse_encoder:
            try:
                sparse_vector = self._sparse_encoder.encode([query])[0]
                sparse_hits = self._spans_v2.search_sparse(sparse_vector, limit, filters=filters_v2)
            except Exception as exc:
                self._log.warning("Sparse retrieval failed: {}", exc)
                sparse_hits = []
            for hit in sparse_hits:
                sparse_scores[hit.event_id] = max(sparse_scores.get(hit.event_id, 0.0), hit.score)
                span_hits.setdefault(hit.event_id, []).append(hit.span_key)

        candidate_ids = set(lexical_scores) | set(dense_scores) | set(sparse_scores)
        if not candidate_ids:
            return self._fallback_ocr_scan(query, time_range, filters, limit)

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
        dense_norm = _normalize_scores(dense_scores)
        sparse_norm = _normalize_scores(sparse_scores)
        now = dt.datetime.now(dt.timezone.utc)

        results: list[RetrievedEvent] = []
        for event in events:
            lex = lexical_norm.get(event.event_id, 0.0)
            dense = dense_norm.get(event.event_id, 0.0)
            sparse = sparse_norm.get(event.event_id, 0.0)
            event_ts = _ensure_aware(event.ts_start)
            age_hours = max((now - event_ts).total_seconds() / 3600, 0.0)
            recency = _recency_bias(age_hours)
            score = _combine_scores(lex, dense, sparse, recency)
            results.append(
                RetrievedEvent(
                    event=event,
                    score=score,
                    matched_span_keys=span_hits.get(event.event_id, []),
                    lexical_score=lex,
                    vector_score=dense,
                    sparse_score=sparse,
                    engine=engine,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results

    def _late_rerank(
        self, query: str, results: list[RetrievedEvent], limit: int
    ) -> list[RetrievedEvent]:
        if not results:
            return results
        if not self._spans_v2 or not self._late_encoder:
            return results
        query_vectors = self._late_encoder.encode_text(query)
        if not query_vectors:
            return results
        try:
            late_hits = self._spans_v2.search_late(query_vectors, limit)
        except Exception as exc:
            self._log.warning("Late retrieval failed: {}", exc)
            return results
        late_scores = {hit.event_id: max(hit.score, 0.0) for hit in late_hits}
        late_norm = _normalize_scores(late_scores)
        reranked: list[RetrievedEvent] = []
        for item in results:
            late = late_norm.get(item.event.event_id, 0.0)
            score = 0.8 * item.score + 0.2 * late
            reranked.append(
                RetrievedEvent(
                    event=item.event,
                    score=score,
                    matched_span_keys=item.matched_span_keys,
                    lexical_score=item.lexical_score,
                    vector_score=item.vector_score,
                    sparse_score=item.sparse_score,
                    late_score=late,
                    engine="late_rerank",
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    def _fallback_ocr_scan(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: RetrieveFilters | None,
        limit: int,
    ) -> list[RetrievedEvent]:
        with self._db.session() as session:
            stmt = select(EventRecord).where(EventRecord.ocr_text.ilike(f"%{query}%"))
            if time_range:
                stmt = stmt.where(EventRecord.ts_start.between(*time_range))
            if filters and filters.apps:
                stmt = stmt.where(EventRecord.app_name.in_(filters.apps))
            if filters and filters.domains:
                stmt = stmt.where(EventRecord.domain.in_(filters.domains))
            stmt = stmt.order_by(EventRecord.ts_start.desc()).limit(limit)
            rows = session.execute(stmt).scalars().all()
        return [RetrievedEvent(event=row, score=0.5, engine="fallback") for row in rows]

    def _persist_trace(
        self, query: str, rewrites: list[str], results: list[RetrievedEvent]
    ) -> None:
        def _write(session) -> None:
            session.add(
                RetrievalTraceRecord(
                    query_text=query,
                    rewrites_json={"rewrites": rewrites},
                    fused_results_json={
                        "results": [
                            {
                                "event_id": result.event.event_id,
                                "score": result.score,
                                "engine": result.engine,
                            }
                            for result in results
                        ]
                    },
                )
            )

        try:
            self._db.transaction(_write)
        except Exception as exc:  # pragma: no cover - best-effort
            self._log.debug("Failed to persist retrieval trace: {}", exc)

    def _rewrite_queries(self, query: str) -> list[str]:
        max_rewrites = self._config.retrieval.fusion_rewrites
        cleaned = _sanitize_query(query)
        tokens = _tokenize_query(cleaned)
        rewrites = [cleaned]
        if tokens:
            rewrites.append(" ".join(tokens))
        if len(tokens) > 1:
            rewrites.append(" AND ".join(tokens))
        if len(tokens) > 2:
            rewrites.append(" ".join(tokens[:3]))
        deduped: list[str] = []
        for rewrite in rewrites:
            trimmed = rewrite.strip()
            if not trimmed:
                continue
            if len(trimmed) > self._config.retrieval.rewrite_max_chars:
                trimmed = trimmed[: self._config.retrieval.rewrite_max_chars]
            if trimmed.lower() in {item.lower() for item in deduped}:
                continue
            deduped.append(trimmed)
            if len(deduped) >= max_rewrites:
                break
        return deduped

    def _rerank_results(self, query: str, results: list[RetrievedEvent]) -> list[RetrievedEvent]:
        if not results:
            return results
        reranker = self._get_reranker()
        if reranker is None:
            return results
        mode = self._runtime.current_mode if self._runtime else None
        if (
            mode == RuntimeMode.FULLSCREEN_HARD_PAUSE
            and self._config.reranker.disable_in_fullscreen
        ):
            return results
        if mode == RuntimeMode.ACTIVE_INTERACTIVE and self._config.reranker.disable_in_active:
            return results
        batch_size = self._config.reranker.batch_size_idle
        device_override = None
        if mode == RuntimeMode.ACTIVE_INTERACTIVE:
            batch_size = self._config.reranker.batch_size_active
            if self._config.reranker.force_cpu_in_active:
                device_override = "cpu"
        if self._runtime:
            profile = self._runtime.qos_profile()
            if profile.reranker_batch_size:
                batch_size = profile.reranker_batch_size
        top_k = min(len(results), self._config.reranker.top_k)
        head = results[:top_k]
        tail = results[top_k:]
        documents = [_build_rerank_document(result.event) for result in head]
        try:
            kwargs: dict[str, object] = {}
            parameters = inspect.signature(reranker.rank).parameters
            if "batch_size" in parameters:
                kwargs["batch_size"] = batch_size
            if "device" in parameters:
                kwargs["device"] = device_override
            scores = reranker.rank(query, documents, **kwargs)
            if len(scores) != len(head):
                raise RuntimeError("Reranker returned mismatched score count")
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_reranker_failure_log > 5.0:
                self._last_reranker_failure_log = now
                self._log.warning("Reranker failed; using hybrid scores: {}", exc)
            return results

        reranked = [
            RetrievedEvent(
                event=item.event,
                score=float(score),
                matched_span_keys=item.matched_span_keys,
                lexical_score=item.lexical_score,
                vector_score=item.vector_score,
                sparse_score=item.sparse_score,
                late_score=item.late_score,
                engine="rerank",
            )
            for item, score in zip(head, scores, strict=True)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked + tail

    def _get_reranker(self) -> CrossEncoderReranker | None:
        if self._config.routing.reranker != "enabled" or not self._config.reranker.enabled:
            return None
        if self._reranker is not None:
            return self._reranker
        if self._reranker_failed:
            return None
        try:
            self._reranker = CrossEncoderReranker(self._config.reranker)
        except Exception as exc:
            self._reranker_failed = True
            self._log.warning("Reranker unavailable; using hybrid scores: {}", exc)
            return None
        return self._reranker


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {key: 1.0 for key in scores}
    return {key: (val - min_score) / (max_score - min_score) for key, val in scores.items()}


def _recency_bias(age_hours: float) -> float:
    decay_hours = 72.0
    return float(math.exp(-age_hours / decay_hours))


def _ensure_aware(timestamp: dt.datetime) -> dt.datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp


def _build_rerank_document(event: EventRecord, max_chars: int = 1000) -> str:
    parts: list[str] = []
    if event.app_name:
        parts.append(f"app: {event.app_name}")
    if event.window_title:
        parts.append(f"title: {event.window_title}")
    url = event.url or event.domain
    if url:
        parts.append(f"url: {url}")
    text = (event.ocr_text or "").strip()
    if text:
        parts.append(f"text: {text}")
    document = " | ".join(parts)
    if len(document) > max_chars:
        return document[:max_chars]
    return document


def _combine_scores(lex: float, dense: float, sparse: float, recency: float) -> float:
    weights = {
        "lex": 0.4,
        "dense": 0.4,
        "sparse": 0.15 if sparse > 0.0 else 0.0,
        "recency": 0.1,
    }
    total = sum(weights.values()) or 1.0
    return (
        weights["lex"] * lex
        + weights["dense"] * dense
        + weights["sparse"] * sparse
        + weights["recency"] * recency
    ) / total


def _retrieval_confidence(results: list[RetrievedEvent]) -> tuple[float, float]:
    if not results:
        return 0.0, 0.0
    top = results[0].score
    second = results[1].score if len(results) > 1 else 0.0
    return top, max(0.0, top - second)


def _is_confident(confidence: tuple[float, float], min_score: float, min_gap: float) -> bool:
    score, gap = confidence
    return score >= min_score and gap >= min_gap


def _assign_ranks(results: list[RetrievedEvent]) -> list[RetrievedEvent]:
    ranked: list[RetrievedEvent] = []
    prev_score: float | None = None
    for idx, item in enumerate(results, start=1):
        rank_gap = 0.0 if prev_score is None else max(0.0, round(prev_score - item.score, 6))
        ranked.append(
            RetrievedEvent(
                event=item.event,
                score=item.score,
                matched_span_keys=item.matched_span_keys,
                lexical_score=item.lexical_score,
                vector_score=item.vector_score,
                sparse_score=item.sparse_score,
                late_score=item.late_score,
                engine=item.engine,
                rank=idx,
                rank_gap=rank_gap,
            )
        )
        prev_score = item.score
    return ranked


def _rrf_fuse(results_lists: list[list[RetrievedEvent]], rrf_k: int) -> list[RetrievedEvent]:
    scores: dict[str, float] = {}
    meta: dict[str, RetrievedEvent] = {}
    for results in results_lists:
        for rank, item in enumerate(results, start=1):
            event_id = item.event.event_id
            scores[event_id] = scores.get(event_id, 0.0) + 1.0 / (rrf_k + rank)
            if event_id not in meta:
                meta[event_id] = item
            else:
                existing = meta[event_id]
                meta[event_id] = RetrievedEvent(
                    event=existing.event,
                    score=existing.score,
                    matched_span_keys=sorted(
                        set(existing.matched_span_keys + item.matched_span_keys)
                    ),
                    lexical_score=max(existing.lexical_score, item.lexical_score),
                    vector_score=max(existing.vector_score, item.vector_score),
                    sparse_score=max(existing.sparse_score, item.sparse_score),
                    late_score=max(existing.late_score, item.late_score),
                    engine="fusion",
                )
    fused: list[RetrievedEvent] = []
    for event_id, score in scores.items():
        base = meta[event_id]
        fused.append(
            RetrievedEvent(
                event=base.event,
                score=score,
                matched_span_keys=base.matched_span_keys,
                lexical_score=base.lexical_score,
                vector_score=base.vector_score,
                sparse_score=base.sparse_score,
                late_score=base.late_score,
                engine="fusion",
            )
        )
    fused.sort(key=lambda item: item.score, reverse=True)
    return fused


def _build_vector_filters(filters: RetrieveFilters | None, *, v2: bool) -> dict | None:
    if not filters:
        return None
    payload_filters: dict[str, object] = {}
    if filters.apps:
        payload_filters["app" if v2 else "app_name"] = list(filters.apps)
    if filters.domains:
        payload_filters["domain"] = list(filters.domains)
    return payload_filters or None


def _sanitize_query(query: str) -> str:
    cleaned = " ".join((query or "").split())
    return re.sub(r"[^A-Za-z0-9_\\-\\s]", " ", cleaned).strip()


def _tokenize_query(query: str) -> list[str]:
    tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", query or "")]
    stopwords = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for"}
    filtered = [token.lower() for token in tokens if token.lower() not in stopwords]
    return filtered or [token.lower() for token in tokens]
