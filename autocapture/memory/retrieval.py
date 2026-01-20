"""Local retrieval utilities for events and evidence."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
import inspect
import math
import os
import time
import re
from pathlib import Path
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
from ..observability.otel import otel_span, record_histogram
from ..storage.database import DatabaseManager
from ..storage.models import (
    CitableSpanRecord,
    EventRecord,
    FrameRecord,
    OCRSpanRecord,
    QueryRecord,
    RetrievalHitRecord,
    RetrievalRunRecord,
    RetrievalTraceRecord,
    TierPlanDecisionRecord,
    TierStatsRecord,
)
from ..contracts_utils import stable_id
from ..runtime_budgets import BudgetManager, BudgetState
from ..storage.ledger import LedgerWriter
from ..time_utils import elapsed_ms, monotonic_now
from .reranker import CrossEncoderReranker
from ..runtime_governor import RuntimeGovernor, RuntimeMode
from .graph_adapters import GraphAdapterGroup


@dataclass(frozen=True)
class RetrieveFilters:
    apps: list[str] | None = None
    domains: list[str] | None = None


@dataclass(frozen=True)
class RetrievalResult:
    event: EventRecord
    score: float
    matched_span_keys: list[str] = field(default_factory=list)
    lexical_score: float = 0.0
    vector_score: float = 0.0
    sparse_score: float = 0.0
    late_score: float = 0.0
    rerank_score: float | None = None
    engine: str = "hybrid"
    rank: int = 0
    rank_gap: float = 0.0
    snippet: str | None = None
    snippet_offset: int | None = None
    bbox: list[int] | None = None
    non_citable: bool = False
    dedupe_group_id: str | None = None
    frame_hash: str | None = None
    frame_id: str | None = None
    event_id: str | None = None


@dataclass(frozen=True)
class RetrievalBatch:
    results: list[RetrievalResult]
    no_evidence: bool = False
    reason: str | None = None
    tier_plan: dict | None = None
    query_id: str | None = None


RetrievedEvent = RetrievalResult


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
        graph_adapters: GraphAdapterGroup | None = None,
        plugin_manager: object | None = None,
    ) -> None:
        self._db = db
        self._config = config or AppConfig()
        self._log = get_logger("retrieval")
        self._ledger = LedgerWriter(self._db)
        self._lexical = LexicalIndex(db)
        if embedder is None:
            if os.environ.get("AUTOCAPTURE_TEST_MODE") or os.environ.get("PYTEST_CURRENT_TEST"):
                if self._config.embed.text_model == "BAAI/bge-base-en-v1.5":
                    self._config.embed.text_model = "local-test"
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
        self._graph_adapters = graph_adapters or GraphAdapterGroup(
            self._config.retrieval.graph_adapters,
            plugin_manager=plugin_manager,
        )

    def retrieve(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: RetrieveFilters | None,
        limit: int = 12,
        offset: int = 0,
        mode: str | None = None,
        request_id: str | None = None,
    ) -> RetrievalBatch:
        query = query.strip()
        if len(query) < 2:
            if time_range:
                results = self._retrieve_time_range(time_range, filters, limit, offset)
                results = self._decorate_results("", results)
                return RetrievalBatch(results=results, no_evidence=not results)
            return RetrievalBatch(results=[], no_evidence=True, reason="query_too_short")
        limit = max(1, limit)
        offset = max(0, offset)
        start_ts = monotonic_now()
        mode_value = (mode or "auto").strip().lower()
        v2_enabled = self._v2_enabled()

        candidate_limit = (limit + offset) * 3
        baseline = self._retrieve_candidates(
            query, time_range, filters, candidate_limit, engine="baseline"
        )
        results = baseline
        rewrites: list[str] = []
        graph_results = self._graph_candidates(query, time_range, filters, candidate_limit)

        enable_fusion = (
            v2_enabled
            and self._config.retrieval.fusion_enabled
            and self._config.retrieval.multi_query_enabled
            and self._config.retrieval.rrf_enabled
            and mode_value in {"auto", "deep"}
        )
        if enable_fusion:
            confidence = _retrieval_confidence(baseline)
            if mode_value == "deep" or not _is_confident(
                confidence,
                self._config.retrieval.fusion_confidence_min,
                self._config.retrieval.fusion_rank_gap_min,
            ):
                rewrites = self._rewrite_queries(query)
                fused_lists = [baseline]
                if graph_results:
                    fused_lists.append(graph_results)
                for rewrite in rewrites:
                    if rewrite.strip().lower() == query.strip().lower():
                        continue
                    fused_lists.append(
                        self._retrieve_candidates(
                            rewrite, time_range, filters, candidate_limit, engine="rewrite"
                        )
                    )
                results = _rrf_fuse(fused_lists, self._config.retrieval.fusion_rrf_k)
        elif graph_results:
            results = _merge_graph_results(results, graph_results)
            results = _assign_ranks(results)

        if v2_enabled and self._config.retrieval.late_enabled and mode_value in {"auto", "deep"}:
            results = self._late_rerank(query, results, candidate_limit)

        results = self._rerank_results(query, results)
        results = _assign_ranks(results)
        if self._config.features.enable_thresholding:
            results = _apply_thresholds(results, self._config.retrieval)
        results = results[offset : offset + limit]
        results = self._decorate_results(query, results)
        latency = elapsed_ms(start_ts)
        retrieval_latency_ms.observe(latency)
        if self._config.retrieval.traces_enabled:
            self._persist_trace(query, rewrites, results)
        if not results:
            return RetrievalBatch(
                results=[],
                no_evidence=True,
                reason="no_results",
            )
        query_class = _classify_query(query, filters)
        query_id = _ensure_query_record(
            self._db,
            query,
            filters,
            query_class,
            BudgetManager(self._config).snapshot(),
        )
        engine_json = {
            "graph": bool(graph_results),
            "lexical": True,
            "vector": True,
            "sparse": bool(self._config.retrieval.sparse_enabled),
            "late": bool(self._config.retrieval.late_enabled),
        }
        self._persist_retrieval_run(
            request_id=request_id,
            query_id=query_id,
            mode=mode_value,
            k=limit,
            result_count=len(results),
            engine_json=engine_json,
        )
        return RetrievalBatch(results=results, no_evidence=False, query_id=query_id)

    def retrieve_tiered(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: RetrieveFilters | None,
        limit: int = 12,
        offset: int = 0,
        mode: str | None = None,
        request_id: str | None = None,
    ) -> RetrievalBatch:
        query = query.strip()
        if len(query) < 2:
            if time_range:
                results = self._retrieve_time_range(time_range, filters, limit, offset)
                results = self._decorate_results("", results)
                return RetrievalBatch(results=results, no_evidence=not results)
            return RetrievalBatch(results=[], no_evidence=True, reason="query_too_short")
        limit = max(1, limit)
        offset = max(0, offset)
        budgets = BudgetManager(self._config)
        budget_state = budgets.start()
        mode_value = (mode or "auto").strip().lower()

        query_class = _classify_query(query, filters)
        query_id = _ensure_query_record(self._db, query, filters, query_class, budgets.snapshot())

        tier_defaults = _load_tier_defaults(self._config)
        fast_defaults = tier_defaults.get("fast", {})
        fusion_defaults = tier_defaults.get("fusion", {})
        rerank_defaults = tier_defaults.get("rerank", {})
        k_lex = max(limit, int(fast_defaults.get("k_lex", limit)))
        k_vec = max(limit, int(fast_defaults.get("k_vec", limit)))
        fusion_k = max(limit, int(fusion_defaults.get("fusion_k", limit)))
        rerank_top_n = max(limit, int(rerank_defaults.get("top_n", limit)))
        rrf_k = int(fusion_defaults.get("rrf_k", self._config.retrieval.fusion_rrf_k))

        candidate_limit = max((limit + offset) * 3, k_lex, k_vec)
        skip_dense = budgets.should_skip_dense(budget_state)
        if skip_dense:
            k_vec = 0
            candidate_limit = budgets.reduce_k(candidate_limit)
            budgets.mark_degraded(budget_state, "retrieve_dense")

        tier_stats = _load_tier_stats(self._db, query_class)
        plan_json, skipped_json, reasons_json = _plan_tiers(
            self._config, mode_value, budgets, budget_state, query_class, tier_stats
        )
        plan_json["tiers_requested"] = list(plan_json.get("tiers", []))
        plan_json["tiers_executed"] = []
        plan_json["skipped_signals"] = {"dense": "budget_low"} if skip_dense else {}
        stage_ms: dict[str, float] = {}
        try:
            self._ledger.append_entry(
                "retrieve_start",
                {
                    "query_id": query_id,
                    "plan": plan_json,
                    "skipped": skipped_json,
                    "reasons": reasons_json,
                },
            )
        except Exception as exc:  # pragma: no cover - best effort
            self._log.debug("Failed to append retrieve_start ledger entry: {}", exc)

        fast_start = time.monotonic()
        baseline = self._retrieve_candidates(
            query,
            time_range,
            filters,
            candidate_limit,
            engine="baseline",
            k_lex=k_lex,
            k_vec=k_vec,
            skip_dense=skip_dense,
        )
        graph_results = self._graph_candidates(query, time_range, filters, candidate_limit)
        fast_ms = (time.monotonic() - fast_start) * 1000
        budgets.record_stage(budget_state, "retrieve_fast", fast_ms)
        stage_ms["FAST"] = fast_ms
        baseline = _assign_ranks(baseline)
        fast_hit_ids = _persist_retrieval_hits(self._db, query_id, "FAST", baseline)
        plan_json["tiers_executed"].append("FAST")

        confidence = _retrieval_confidence(baseline)
        if "FUSION" in plan_json.get("tiers", []) and mode_value != "deep":
            if _is_confident(
                confidence,
                self._config.retrieval.fusion_confidence_min,
                self._config.retrieval.fusion_rank_gap_min,
            ):
                plan_json["tiers"].remove("FUSION")
                skipped_json["tiers"].append("FUSION")
                reasons_json["tiers"]["FUSION"] = "confidence_strong"

        if "RERANK" not in plan_json.get("tiers", []) and query_class == "FACT_NUMERIC_TIMEBOUND":
            if _has_numeric_candidate(baseline) and not budgets.should_skip_rerank(budget_state):
                if "RERANK" in skipped_json.get("tiers", []):
                    skipped_json["tiers"].remove("RERANK")
                plan_json["tiers"].append("RERANK")
                reasons_json["tiers"]["RERANK"] = "numeric_guard"

        results = baseline
        rewrites: list[str] = []
        fusion_hit_ids: list[str] = []
        if "FUSION" in plan_json.get("tiers", []):
            should_fuse = mode_value == "deep" or not _is_confident(
                confidence,
                self._config.retrieval.fusion_confidence_min,
                self._config.retrieval.fusion_rank_gap_min,
            )
            if should_fuse:
                fusion_start = time.monotonic()
                rewrites = self._rewrite_queries(query)
                fused_lists = [baseline]
                if graph_results:
                    fused_lists.append(graph_results)
                for rewrite in rewrites:
                    if rewrite.strip().lower() == query.strip().lower():
                        continue
                    fused_lists.append(
                        self._retrieve_candidates(
                            rewrite,
                            time_range,
                            filters,
                            candidate_limit,
                            engine="rewrite",
                            k_lex=k_lex,
                            k_vec=k_vec,
                            skip_dense=skip_dense,
                        )
                    )
                results = _rrf_fuse(fused_lists, rrf_k)
                if fusion_k > 0:
                    results = results[:fusion_k]
                fusion_ms = (time.monotonic() - fusion_start) * 1000
                budgets.record_stage(
                    budget_state,
                    "retrieve_fusion",
                    fusion_ms,
                )
                stage_ms["FUSION"] = fusion_ms
                results = _assign_ranks(results)
                fusion_hit_ids = _persist_retrieval_hits(self._db, query_id, "FUSION", results)
                plan_json["tiers_executed"].append("FUSION")
            else:
                skipped_json["tiers"].append("FUSION")
                reasons_json["tiers"]["FUSION"] = "confidence_strong"
                fusion_hit_ids = []
        elif graph_results:
            results = _merge_graph_results(results, graph_results)

        if "RERANK" in plan_json.get("tiers", []):
            if budgets.should_skip_rerank(budget_state):
                skipped_json["tiers"].append("RERANK")
                reasons_json["tiers"]["RERANK"] = "budget_low"
                budgets.mark_degraded(budget_state, "retrieve_rerank")
                rerank_hit_ids = []
            else:
                rerank_start = time.monotonic()
                results = self._rerank_results(query, results, top_k=rerank_top_n)
                rerank_ms = (time.monotonic() - rerank_start) * 1000
                budgets.record_stage(
                    budget_state,
                    "retrieve_rerank",
                    rerank_ms,
                )
                stage_ms["RERANK"] = rerank_ms
                results = _assign_ranks(results)
                rerank_hit_ids = _persist_retrieval_hits(self._db, query_id, "RERANK", results)
                plan_json["tiers_executed"].append("RERANK")
        else:
            rerank_hit_ids = []

        if self._config.retrieval.traces_enabled:
            self._persist_trace(query, rewrites, results)

        try:
            self._ledger.append_entry(
                "retrieve_done",
                {
                    "query_id": query_id,
                    "hit_ids": {
                        "fast": fast_hit_ids,
                        "fusion": fusion_hit_ids,
                        "rerank": rerank_hit_ids,
                    },
                },
            )
        except Exception as exc:  # pragma: no cover - best effort
            self._log.debug("Failed to append retrieve_done ledger entry: {}", exc)

        plan_json["stage_ms"] = stage_ms
        _persist_tier_plan(
            self._db, query_id, plan_json, skipped_json, reasons_json, budgets.snapshot()
        )

        results = results[offset : offset + limit]
        results = self._decorate_results(query, results)
        if not results:
            return RetrievalBatch(
                results=[],
                no_evidence=True,
                reason="no_results",
                tier_plan=plan_json,
                query_id=query_id,
            )
        engine_json = {
            "graph": bool(graph_results),
            "lexical": True,
            "vector": True,
            "sparse": bool(self._config.retrieval.sparse_enabled),
            "late": bool(self._config.retrieval.late_enabled),
            "tier_plan": plan_json,
        }
        self._persist_retrieval_run(
            request_id=request_id,
            query_id=query_id,
            mode=mode_value,
            k=limit,
            result_count=len(results),
            engine_json=engine_json,
        )
        return RetrievalBatch(
            results=results,
            no_evidence=False,
            tier_plan=plan_json,
            query_id=query_id,
        )

    def _v2_enabled(self) -> bool:
        config = self._config.retrieval
        return bool(
            config.v2_enabled
            or config.use_spans_v2
            or config.sparse_enabled
            or config.late_enabled
            or config.fusion_enabled
        )

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
        return [
            RetrievedEvent(
                event=row,
                score=0.4,
                event_id=row.event_id,
                frame_id=row.event_id,
                frame_hash=getattr(row, "frame_hash", None) or row.screenshot_hash,
            )
            for row in rows
        ]

    def _retrieve_candidates(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: RetrieveFilters | None,
        limit: int,
        *,
        engine: str,
        k_lex: int | None = None,
        k_vec: int | None = None,
        skip_dense: bool = False,
    ) -> list[RetrievedEvent]:
        lexical_start = time.monotonic()
        with otel_span("retrieval.lexical", {"stage_name": "retrieval.lexical"}):
            lexical_hits = self._lexical.search(query, limit=k_lex or limit)
        record_histogram(
            "index_lexical_ms",
            (time.monotonic() - lexical_start) * 1000,
            {"stage_name": "index_lexical"},
        )
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

        if dense_vector is not None and not skip_dense:
            try:
                vector_limit = k_vec or limit
                if self._spans_v2 and self._config.retrieval.use_spans_v2:
                    vector_start = time.monotonic()
                    with otel_span("retrieval.vector", {"stage_name": "retrieval.vector"}):
                        dense_hits = self._spans_v2.search_dense(
                            dense_vector,
                            vector_limit,
                            filters=filters_v2,
                            embedding_model=self._embedder.model_name,
                        )
                    record_histogram(
                        "vector_search_ms",
                        (time.monotonic() - vector_start) * 1000,
                        {"stage_name": "vector_search"},
                    )
                else:
                    vector_start = time.monotonic()
                    with otel_span("retrieval.vector", {"stage_name": "retrieval.vector"}):
                        dense_hits = self._vector.search(
                            dense_vector,
                            vector_limit,
                            filters=filters_map,
                            embedding_model=self._embedder.model_name,
                        )
                    record_histogram(
                        "vector_search_ms",
                        (time.monotonic() - vector_start) * 1000,
                        {"stage_name": "vector_search"},
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
        if (
            not skip_dense
            and self._config.retrieval.sparse_enabled
            and self._spans_v2
            and self._sparse_encoder
        ):
            try:
                sparse_vector = self._sparse_encoder.encode([query])[0]
                sparse_hits = self._spans_v2.search_sparse(
                    sparse_vector, k_vec or limit, filters=filters_v2
                )
            except Exception as exc:
                self._log.warning("Sparse retrieval failed: {}", exc)
                sparse_hits = []
            for hit in sparse_hits:
                sparse_scores[hit.event_id] = max(sparse_scores.get(hit.event_id, 0.0), hit.score)
                span_hits.setdefault(hit.event_id, []).append(hit.span_key)

        late_stage1_scores: dict[str, float] = {}
        if (
            not skip_dense
            and self._config.retrieval.late_stage1_enabled
            and self._config.retrieval.late_enabled
            and self._spans_v2
            and self._late_encoder
            and _late_stage1_window_ok(time_range, self._config.retrieval.late_stage1_max_days)
        ):
            query_vectors = self._late_encoder.encode_text(query)
            if query_vectors:
                try:
                    late_hits = self._spans_v2.search_late(
                        query_vectors,
                        self._config.retrieval.late_stage1_k,
                        filters=filters_v2,
                    )
                except Exception as exc:
                    self._log.warning("Late stage-1 retrieval failed: {}", exc)
                    late_hits = []
                for hit in late_hits:
                    late_stage1_scores[hit.event_id] = max(
                        late_stage1_scores.get(hit.event_id, 0.0), hit.score
                    )
                    span_hits.setdefault(hit.event_id, []).append(hit.span_key)

        candidate_ids = (
            set(lexical_scores) | set(dense_scores) | set(sparse_scores) | set(late_stage1_scores)
        )
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

        late_stage1_norm = _normalize_scores(late_stage1_scores)
        results: list[RetrievedEvent] = []
        for event in events:
            lex = lexical_norm.get(event.event_id, 0.0)
            dense = dense_norm.get(event.event_id, 0.0)
            sparse = sparse_norm.get(event.event_id, 0.0)
            late_stage1 = late_stage1_norm.get(event.event_id, 0.0)
            event_ts = _ensure_aware(event.ts_start)
            age_hours = max((now - event_ts).total_seconds() / 3600, 0.0)
            recency = _recency_bias(age_hours)
            score = _combine_scores(lex, dense, sparse, recency, late_stage1)
            results.append(
                RetrievedEvent(
                    event=event,
                    score=score,
                    matched_span_keys=sorted(set(span_hits.get(event.event_id, []))),
                    lexical_score=lex,
                    vector_score=dense,
                    sparse_score=sparse,
                    late_score=late_stage1,
                    engine=engine,
                    frame_hash=getattr(event, "frame_hash", None) or event.screenshot_hash,
                    frame_id=event.event_id,
                    event_id=event.event_id,
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
        if not late_hits:
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
                    frame_hash=item.frame_hash,
                    frame_id=item.frame_id,
                    event_id=item.event_id,
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
        return [
            RetrievedEvent(
                event=row,
                score=0.4,
                lexical_score=0.4,
                engine="fallback",
                frame_hash=getattr(row, "frame_hash", None) or row.screenshot_hash,
                frame_id=row.event_id,
                event_id=row.event_id,
            )
            for row in rows
        ]

    def _persist_trace(
        self, query: str, rewrites: list[str], results: list[RetrievedEvent]
    ) -> None:
        rewrites_payload = _sorted_json({"rewrites": rewrites})
        fused_payload = _sorted_json(
            {
                "results": [
                    {
                        "event_id": result.event.event_id,
                        "score": result.score,
                        "engine": result.engine,
                    }
                    for result in results
                ]
            }
        )

        def _write(session) -> None:
            session.add(
                RetrievalTraceRecord(
                    query_text=query,
                    rewrites_json=rewrites_payload,
                    fused_results_json=fused_payload,
                )
            )

        try:
            self._db.transaction(_write)
        except Exception as exc:  # pragma: no cover - best-effort
            self._log.debug("Failed to persist retrieval trace: {}", exc)

    def _persist_retrieval_run(
        self,
        *,
        request_id: str | None,
        query_id: str | None,
        mode: str,
        k: int,
        result_count: int,
        engine_json: dict,
    ) -> None:
        run_id = stable_id(
            "retrieval_run",
            {
                "query_id": query_id,
                "mode": mode,
                "k": k,
                "result_count": result_count,
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )

        def _write(session) -> None:
            if session.get(RetrievalRunRecord, run_id):
                return
            session.add(
                RetrievalRunRecord(
                    run_id=run_id,
                    request_id=request_id,
                    query_id=query_id,
                    mode=mode or "auto",
                    k=int(k),
                    result_count=int(result_count),
                    engine_json=engine_json or {},
                    created_at=dt.datetime.now(dt.timezone.utc),
                )
            )

        try:
            self._db.transaction(_write)
        except Exception:
            return

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

    def _rerank_results(
        self, query: str, results: list[RetrievedEvent], *, top_k: int | None = None
    ) -> list[RetrievedEvent]:
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
            budget = self._runtime.qos_budget()
            if budget.gpu_policy in {"prefer_cpu", "disallow_gpu"}:
                device_override = "cpu"
        limit = min(len(results), self._config.reranker.top_k)
        if top_k is not None:
            limit = min(limit, int(top_k))
        head = results[:limit]
        tail = results[limit:]
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
                rerank_score=float(score),
                engine="rerank",
                frame_hash=item.frame_hash,
                frame_id=item.frame_id,
                event_id=item.event_id,
            )
            for item, score in zip(head, scores, strict=True)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked + tail

    def _decorate_results(self, query: str, results: list[RetrievedEvent]) -> list[RetrievedEvent]:
        if not results:
            return results
        event_ids = [result.event.event_id for result in results]
        spans_by_event = _load_spans(self._db, event_ids)
        decorated: list[RetrievedEvent] = []
        for item in results:
            event = item.event
            text = event.ocr_text or ""
            snippet, offset = _snippet_for_query(text, query)
            spans = spans_by_event.get(event.event_id, [])
            matched = _select_span(spans, item.matched_span_keys, query)
            bbox = _span_bbox(matched.bbox) if matched is not None else None
            non_citable = bool(item.non_citable)
            if matched is None or bbox is None:
                non_citable = True
            decorated.append(
                RetrievedEvent(
                    event=event,
                    score=item.score,
                    matched_span_keys=item.matched_span_keys,
                    lexical_score=item.lexical_score,
                    vector_score=item.vector_score,
                    sparse_score=item.sparse_score,
                    late_score=item.late_score,
                    rerank_score=item.rerank_score,
                    engine=item.engine,
                    rank=item.rank,
                    rank_gap=item.rank_gap,
                    snippet=snippet,
                    snippet_offset=offset if snippet else None,
                    bbox=bbox,
                    non_citable=non_citable,
                    dedupe_group_id=item.dedupe_group_id,
                    frame_hash=item.frame_hash,
                    frame_id=item.frame_id or event.event_id,
                    event_id=item.event_id or event.event_id,
                )
            )
        return decorated

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

    def _graph_candidates(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: RetrieveFilters | None,
        limit: int,
    ) -> list[RetrievedEvent]:
        if not self._graph_adapters or not self._graph_adapters.enabled():
            return []
        if not query:
            return []
        time_payload = (
            (time_range[0].isoformat(), time_range[1].isoformat()) if time_range else None
        )
        filter_payload: dict[str, list[str]] = {}
        if filters and filters.apps:
            filter_payload["apps"] = list(filters.apps)
        if filters and filters.domains:
            filter_payload["domains"] = list(filters.domains)
        with otel_span("retrieval.graph", {"stage_name": "retrieval.graph"}):
            hits = self._graph_adapters.query(
                query,
                limit=min(limit, 100),
                time_range=time_payload,
                filters=filter_payload,
            )
        if not hits:
            return []
        scores: dict[str, float] = {}
        snippets: dict[str, str | None] = {}
        sources: dict[str, str] = {}
        for hit in hits:
            if not hit.event_id:
                continue
            score = max(float(hit.score), 0.0)
            current = scores.get(hit.event_id)
            if current is None or score > current:
                scores[hit.event_id] = score
                snippets[hit.event_id] = hit.snippet
                sources[hit.event_id] = hit.source
        if not scores:
            return []
        with self._db.session() as session:
            stmt = select(EventRecord).where(EventRecord.event_id.in_(scores))
            if time_range:
                stmt = stmt.where(EventRecord.ts_start.between(*time_range))
            if filters and filters.apps:
                stmt = stmt.where(EventRecord.app_name.in_(filters.apps))
            if filters and filters.domains:
                stmt = stmt.where(EventRecord.domain.in_(filters.domains))
            events = session.execute(stmt).scalars().all()
        results: list[RetrievedEvent] = []
        for event in events:
            event_id = event.event_id
            score = scores.get(event_id, 0.0)
            results.append(
                RetrievedEvent(
                    event=event,
                    score=score,
                    matched_span_keys=[],
                    lexical_score=0.0,
                    vector_score=0.0,
                    sparse_score=0.0,
                    late_score=0.0,
                    rerank_score=None,
                    engine=f"graph:{sources.get(event_id, 'graph')}",
                    snippet=snippets.get(event_id),
                    frame_hash=getattr(event, "frame_hash", None) or event.screenshot_hash,
                    frame_id=event.event_id,
                    event_id=event.event_id,
                )
            )
        results.sort(key=lambda item: (-item.score, item.event.event_id))
        return results


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


def _late_stage1_window_ok(
    time_range: tuple[dt.datetime, dt.datetime] | None, max_days: int
) -> bool:
    if not time_range:
        return False
    start, end = time_range
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)
    delta_days = abs((end - start).total_seconds()) / 86400.0
    return delta_days <= max(1, int(max_days))


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


def _combine_scores(
    lex: float, dense: float, sparse: float, recency: float, late: float = 0.0
) -> float:
    weights = {
        "lex": 0.4,
        "dense": 0.4,
        "sparse": 0.15 if sparse > 0.0 else 0.0,
        "late": 0.1 if late > 0.0 else 0.0,
        "recency": 0.1,
    }
    total = sum(weights.values()) or 1.0
    return (
        weights["lex"] * lex
        + weights["dense"] * dense
        + weights["sparse"] * sparse
        + weights["late"] * late
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
                rerank_score=item.rerank_score,
                engine=item.engine,
                rank=idx,
                rank_gap=rank_gap,
                frame_hash=item.frame_hash,
                frame_id=item.frame_id,
                event_id=item.event_id,
            )
        )
        prev_score = item.score
    return ranked


def _rrf_fuse(results_lists: list[list[RetrievedEvent]], rrf_k: int) -> list[RetrievedEvent]:
    scores: dict[str, float] = {}
    best_ranks: dict[str, int] = {}
    meta: dict[str, RetrievedEvent] = {}
    graph_only: dict[str, bool] = {}
    for results in results_lists:
        for rank, item in enumerate(results, start=1):
            event_id = item.event.event_id
            scores[event_id] = scores.get(event_id, 0.0) + 1.0 / (rrf_k + rank)
            best_ranks[event_id] = min(best_ranks.get(event_id, rank), rank)
            is_graph = (item.engine or "").startswith("graph")
            graph_only[event_id] = graph_only.get(event_id, True) and is_graph
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
                    rerank_score=existing.rerank_score,
                    engine="fusion",
                    frame_hash=existing.frame_hash,
                    frame_id=existing.frame_id,
                    event_id=existing.event_id,
                )
    fused: list[RetrievedEvent] = []
    for event_id, score in scores.items():
        base = meta[event_id]
        engine = base.engine if graph_only.get(event_id, False) else "fusion"
        fused.append(
            RetrievedEvent(
                event=base.event,
                score=score,
                matched_span_keys=base.matched_span_keys,
                lexical_score=base.lexical_score,
                vector_score=base.vector_score,
                sparse_score=base.sparse_score,
                late_score=base.late_score,
                rerank_score=base.rerank_score,
                engine=engine,
                frame_hash=base.frame_hash,
                frame_id=base.frame_id,
                event_id=base.event_id,
            )
        )
    fused.sort(
        key=lambda item: (
            -item.score,
            best_ranks.get(item.event.event_id, 10_000),
            item.event.event_id,
        )
    )
    return fused


def _merge_graph_results(
    primary: list[RetrievedEvent], graph_results: list[RetrievedEvent]
) -> list[RetrievedEvent]:
    if not graph_results:
        return primary
    if not primary:
        return graph_results
    primary_ids = {item.event.event_id for item in primary}
    merged = list(primary)
    for item in graph_results:
        if item.event.event_id in primary_ids:
            continue
        merged.append(item)
    merged.sort(key=lambda item: (-item.score, item.event.event_id))
    return merged


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


def _apply_thresholds(results: list[RetrievedEvent], config) -> list[RetrievedEvent]:
    if not results:
        return results
    filtered: list[RetrievedEvent] = []
    for item in results:
        if (item.engine or "").startswith("graph"):
            filtered.append(item)
            continue
        checks: list[bool] = []
        if item.lexical_score is not None:
            checks.append(item.lexical_score >= config.lexical_min_score)
        if item.vector_score is not None:
            checks.append(item.vector_score >= config.dense_min_score)
        if item.rerank_score is not None:
            checks.append(item.rerank_score >= config.rerank_min_score)
        if item.sparse_score is not None:
            checks.append(item.sparse_score >= getattr(config, "sparse_min_score", 0.0))
        if item.late_score is not None:
            checks.append(item.late_score >= getattr(config, "late_min_score", 0.0))
        if checks and not any(checks):
            continue
        filtered.append(item)
    return filtered


def _load_spans(db: DatabaseManager, event_ids: list[str]) -> dict[str, list[OCRSpanRecord]]:
    if not event_ids:
        return {}
    with db.session() as session:
        rows = (
            session.execute(
                select(OCRSpanRecord)
                .where(OCRSpanRecord.capture_id.in_(event_ids))
                .order_by(OCRSpanRecord.start.asc())
            )
            .scalars()
            .all()
        )
    spans_by_event: dict[str, list[OCRSpanRecord]] = {}
    for row in rows:
        spans_by_event.setdefault(row.capture_id, []).append(row)
    return spans_by_event


def _select_span(
    spans: list[OCRSpanRecord], matched_keys: list[str], query: str
) -> OCRSpanRecord | None:
    if not spans:
        return None
    matched_set = {str(key) for key in matched_keys if key}
    if matched_set:
        for span in spans:
            if str(span.span_key) in matched_set:
                return span
    lowered = (query or "").lower().strip()
    if lowered:
        for span in spans:
            if lowered in (span.text or "").lower():
                return span
    return spans[0]


def _span_bbox(raw: object) -> list[int] | None:
    if raw is None:
        return None
    coords: list[float] = []
    if isinstance(raw, dict):
        for key in ("x0", "y0", "x1", "y1"):
            value = raw.get(key)
            if value is None:
                return None
            try:
                coords.append(float(value))
            except (TypeError, ValueError):
                return None
    elif isinstance(raw, list):
        coords = [float(val) for val in raw if isinstance(val, (int, float))]
    else:
        return None
    if len(coords) >= 8:
        xs = coords[0::2]
        ys = coords[1::2]
        if not xs or not ys:
            return None
        x0, x1 = int(min(xs)), int(max(xs))
        y0, y1 = int(min(ys)), int(max(ys))
        return [x0, y0, x1, y1]
    if len(coords) >= 4:
        x0, y0, x1, y1 = [int(val) for val in coords[:4]]
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    return None


def _sorted_json(payload: dict) -> dict:
    if not payload:
        return {}
    return json.loads(json.dumps(payload, sort_keys=True))


def _snippet_for_query(text: str, query: str, window: int = 200) -> tuple[str, int]:
    if not text:
        return "", 0
    lower = text.lower()
    q = (query or "").lower()
    idx = lower.find(q) if q else -1
    if idx == -1:
        return text[: min(400, len(text))], 0
    start = max(idx - window, 0)
    end = min(idx + len(q) + window, len(text))
    return text[start:end], start


def _classify_query(query: str, filters: RetrieveFilters | None) -> str:
    lowered = (query or "").lower()
    has_digit = any(ch.isdigit() for ch in lowered)
    has_time = any(
        token in lowered
        for token in (
            "today",
            "yesterday",
            "tomorrow",
            "last",
            "week",
            "month",
            "year",
            "hour",
            "minute",
            "day",
        )
    )
    if has_digit or has_time:
        return "FACT_NUMERIC_TIMEBOUND"
    tokens = re.findall(r"[a-z0-9]+", lowered)
    if filters and (filters.apps or filters.domains):
        return "FILTERED"
    if len(tokens) <= 2:
        return "SHORT"
    return "GENERAL"


def _ensure_query_record(
    db: DatabaseManager,
    query: str,
    filters: RetrieveFilters | None,
    query_class: str,
    budget_snapshot,
) -> str:
    filters_json = {
        "apps": filters.apps if filters else None,
        "domains": filters.domains if filters else None,
    }
    query_id = stable_id(
        "query",
        {"query": query, "filters": filters_json, "query_class": query_class},
    )

    def _write(session):
        existing = session.get(QueryRecord, query_id)
        if existing:
            return query_id
        session.add(
            QueryRecord(
                query_id=query_id,
                query_text=query,
                normalized_text=query.lower().strip(),
                filters_json=filters_json,
                query_class=query_class,
                budgets_json={
                    "total_ms": budget_snapshot.total_ms,
                    "stages": budget_snapshot.stages,
                    "degrade": budget_snapshot.degrade,
                },
                created_at=dt.datetime.now(dt.timezone.utc),
            )
        )
        return query_id

    return db.transaction(_write)


def _plan_tiers(
    config: AppConfig,
    mode: str,
    budgets: BudgetManager,
    state: BudgetState,
    query_class: str,
    tier_stats: dict[str, TierStatsRecord],
):
    plan = {"tiers": ["FAST"]}
    skipped = {"tiers": []}
    reasons = {"tiers": {}}

    fusion_reason = _tier_skip_reason(
        tier_stats, "FUSION", config, budgets, state, stage="retrieve_fusion"
    )
    if config.retrieval.fusion_enabled and mode in {"auto", "deep"}:
        if fusion_reason:
            skipped["tiers"].append("FUSION")
            reasons["tiers"]["FUSION"] = fusion_reason
        else:
            plan["tiers"].append("FUSION")

    rerank_reason = None
    if config.routing.reranker == "enabled" and config.reranker.enabled:
        if budgets.should_skip_rerank(state) and query_class != "FACT_NUMERIC_TIMEBOUND":
            rerank_reason = "budget_low"
        else:
            rerank_reason = _tier_skip_reason(
                tier_stats, "RERANK", config, budgets, state, stage="retrieve_rerank"
            )
        if rerank_reason and query_class != "FACT_NUMERIC_TIMEBOUND":
            skipped["tiers"].append("RERANK")
            reasons["tiers"]["RERANK"] = rerank_reason
        else:
            plan["tiers"].append("RERANK")
    return plan, skipped, reasons


def _tier_skip_reason(
    tier_stats: dict[str, TierStatsRecord],
    tier: str,
    config: AppConfig,
    budgets: BudgetManager,
    state: BudgetState,
    *,
    stage: str,
) -> str | None:
    record = tier_stats.get(tier)
    if record and record.window_n and record.window_n >= config.next10.tier_stats_window:
        help_rate = float(record.help_rate or 0.0)
        if help_rate < config.next10.tier_help_rate_min:
            remaining = budgets.remaining_ms(state)
            p95_ms = float(record.p95_ms or 0.0)
            threshold = max(float(budgets.budget_ms(stage)), remaining * 0.6)
            if p95_ms and p95_ms > threshold:
                return "low_help_rate_high_latency"
    if budgets.budget_ms(stage) and budgets.remaining_ms(state) < budgets.budget_ms(stage):
        return "budget_low"
    return None


def _load_tier_stats(db: DatabaseManager, query_class: str) -> dict[str, TierStatsRecord]:
    if not query_class:
        return {}
    with db.session() as session:
        rows = (
            session.execute(
                select(TierStatsRecord).where(TierStatsRecord.query_class == query_class)
            )
            .scalars()
            .all()
        )
    return {row.tier: row for row in rows}


def _load_tier_defaults(config: AppConfig) -> dict:
    path = Path(config.next10.tiers_defaults_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _has_numeric_candidate(results: list[RetrievedEvent]) -> bool:
    if not results:
        return False
    pattern = re.compile(r"\b\d+(?:[.,]\d+)?\b")
    for item in results:
        text = item.snippet or item.event.ocr_text or ""
        if pattern.search(text):
            return True
    return False


def _persist_tier_plan(
    db: DatabaseManager,
    query_id: str,
    plan_json: dict,
    skipped_json: dict,
    reasons_json: dict,
    budget_snapshot,
) -> None:
    decision_id = stable_id(
        "tier_plan",
        {"query_id": query_id, "plan": plan_json, "skipped": skipped_json},
    )

    def _write(session):
        existing = session.get(TierPlanDecisionRecord, decision_id)
        if existing:
            return
        session.add(
            TierPlanDecisionRecord(
                decision_id=decision_id,
                query_id=query_id,
                plan_json=plan_json,
                skipped_json=skipped_json,
                reasons_json=reasons_json,
                budgets_json={
                    "total_ms": budget_snapshot.total_ms,
                    "stages": budget_snapshot.stages,
                    "degrade": budget_snapshot.degrade,
                },
                schema_version=1,
                created_at=dt.datetime.now(dt.timezone.utc),
            )
        )

    db.transaction(_write)


def _persist_retrieval_hits(
    db: DatabaseManager, query_id: str, tier: str, results: list[RetrievedEvent]
) -> list[str]:
    if not results:
        return []
    event_ids = [item.event.event_id for item in results]
    hit_ids: list[str] = []

    def _write(session) -> None:
        spans = (
            session.execute(
                select(CitableSpanRecord).where(CitableSpanRecord.frame_id.in_(event_ids))
            )
            .scalars()
            .all()
        )
        frames = {
            row.frame_id: row
            for row in session.execute(
                select(FrameRecord).where(FrameRecord.frame_id.in_(event_ids))
            )
            .scalars()
            .all()
        }
        spans_by_event: dict[str, list[CitableSpanRecord]] = {}
        for span in spans:
            spans_by_event.setdefault(span.frame_id, []).append(span)
        for span_list in spans_by_event.values():
            span_list.sort(key=lambda s: (s.legacy_span_key or "", s.span_id))

        for item in results:
            event_id = item.event.event_id
            span_id = _select_span_id(spans_by_event.get(event_id, []), item.matched_span_keys)
            span_record = None
            if span_id:
                for span in spans_by_event.get(event_id, []):
                    if span.span_id == span_id:
                        span_record = span
                        break
            citable = _is_citable_span(span_record, frames.get(event_id))
            scores_json = {
                "lexical": item.lexical_score,
                "dense": item.vector_score,
                "sparse": item.sparse_score,
                "late": item.late_score,
                "rerank": item.rerank_score,
                "engine": item.engine,
            }
            hit_id = stable_id(
                "hit",
                {
                    "query_id": query_id,
                    "tier": tier,
                    "event_id": event_id,
                    "span_id": span_id,
                    "rank": item.rank,
                },
            )
            existing = session.get(RetrievalHitRecord, hit_id)
            if existing:
                hit_ids.append(hit_id)
                continue
            session.add(
                RetrievalHitRecord(
                    hit_id=hit_id,
                    query_id=query_id,
                    tier=tier,
                    span_id=span_id,
                    event_id=event_id,
                    score=float(item.score),
                    rank=int(item.rank),
                    scores_json=scores_json,
                    citable=citable,
                    schema_version=1,
                    created_at=dt.datetime.now(dt.timezone.utc),
                )
            )
            hit_ids.append(hit_id)

    db.transaction(_write)
    return hit_ids


def _select_span_id(spans: list[CitableSpanRecord], matched_span_keys: list[str]) -> str | None:
    if not spans:
        return None
    matched_set = {str(key) for key in matched_span_keys if key}
    if matched_set:
        for span in spans:
            if span.legacy_span_key and span.legacy_span_key in matched_set:
                return span.span_id
    return spans[0].span_id if spans else None


def _is_citable_span(span: CitableSpanRecord | None, frame: FrameRecord | None) -> bool:
    if span is None or span.tombstoned:
        return False
    if span.bbox is None:
        return False
    if frame is None or not frame.media_path:
        return False
    return True
