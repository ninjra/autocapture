"""Agentic answering pipeline (LangGraph optional)."""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from types import SimpleNamespace

from sqlalchemy import select
from ..config import AppConfig
from ..logging_utils import get_logger
from ..answer.coverage import coverage_metrics, extract_sentence_citations
from ..answer.claims import parse_claims_json, render_claims_answer
from ..answer.claim_validation import ClaimValidator, EvidenceLineInfo
from ..answer.entailment import heuristic_entailment, judge_entailment
from ..answer.conflict import detect_conflicts
from ..answer.integrity import check_citations
from ..answer.no_evidence import build_no_evidence_payload
from ..answer.provenance import append_provenance_chain, verify_provenance
from ..memory.compression import CompressedAnswer, extractive_answer
from ..memory.context_pack import (
    EvidenceItem,
    EvidenceSpan,
    build_context_pack,
    build_evidence_payload,
)
from ..memory.prompt_injection import scan_prompt_injection
from ..memory.time_intent import (
    is_time_only_expression,
    resolve_time_intent,
    resolve_time_range_for_query,
    resolve_timezone,
)
from ..memory.threads import ThreadRetrievalService
from ..memory.retrieval import (
    RetrieveFilters,
    RetrievalService,
    _classify_query,
    _ensure_query_record,
)
from ..memory.tier_stats import update_tier_stats
from ..model_ops import StageRouter
from ..storage.models import (
    AnswerCitationRecord,
    AnswerRecord,
    AnswerClaimRecord,
    AnswerClaimCitationRecord,
    EvidenceLineMapRecord,
    ProviderCallRecord,
    RequestRunRecord,
    StageRunRecord,
    ProviderHealthRecord,
    EvidenceItemRecord,
    ClaimRecord,
    ClaimCitationRecord,
    CitableSpanRecord,
    EventRecord,
)
from ..storage.ledger import LedgerWriter
from ..contracts_utils import stable_id, sha256_text
from ..runtime_budgets import BudgetManager
from ..memory.verification import Claim, RulesVerifier
from ..observability.otel import otel_span, record_histogram
from ..observability.metrics import verification_failures_total
from ..plugins import PluginManager


@dataclass
class AnswerGraphResult:
    answer: str
    citations: list[str]
    context_pack: dict
    warnings: list[str]
    used_llm: bool
    mode: str = "NORMAL"
    coverage: dict | None = None
    confidence: dict | None = None
    budgets: dict | None = None
    degraded_stages: list[str] | None = None
    hints: list[dict] | None = None
    actions: list[dict] | None = None
    conflict_summary: dict | None = None
    answer_id: str | None = None
    query_id: str | None = None
    response_json: dict | None = None
    response_tron: str | None = None
    context_pack_tron: str | None = None
    prompt_strategy: dict | None = None


@dataclass(frozen=True)
class RefinedQueryIntent:
    refined_query: str | None
    time_expression: str | None
    time_range_payload: dict[str, str] | None


class AnswerGraph:
    def __init__(
        self,
        config: AppConfig,
        retrieval: RetrievalService,
        *,
        prompt_registry,
        entities,
        plugin_manager: PluginManager | None = None,
    ) -> None:
        self._config = config
        self._retrieval = retrieval
        self._prompt_registry = prompt_registry
        self._entities = entities
        self._log = get_logger("answer_graph")
        self._ledger = LedgerWriter(retrieval._db)  # type: ignore[attr-defined]
        self._plugins = plugin_manager or PluginManager(config)
        self._stage_router = StageRouter(config, plugin_manager=self._plugins)
        self._thread_retrieval = ThreadRetrievalService(
            config,
            retrieval._db,  # type: ignore[attr-defined]
            embedder=getattr(retrieval, "_embedder", None),
            vector_index=getattr(retrieval, "_vector", None),
        )
        self._compressor = None
        self._verifier = None
        self._claim_validator = ClaimValidator(config.verification.citation_validator)

    async def run(
        self,
        query: str,
        *,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: dict[str, list[str]] | None,
        k: int,
        sanitized: bool,
        extractive_only: bool,
        routing: dict[str, str],
        routing_override: str | None = None,
        aggregates: dict | None = None,
        output_format: str = "text",
        context_pack_format: str = "json",
    ) -> AnswerGraphResult:
        warnings: list[str] = []
        request_id = stable_id(
            "request_run",
            {"query": query, "ts": dt.datetime.now(dt.timezone.utc).isoformat()},
        )
        request_started_at = dt.datetime.now(dt.timezone.utc)
        if hasattr(self._retrieval, "_db"):
            self._retrieval._db.transaction(  # type: ignore[attr-defined]
                lambda session: session.add(
                    RequestRunRecord(
                        request_id=request_id,
                        query_id=None,
                        query_text=query,
                        status="started",
                        warnings_json={},
                        started_at=request_started_at,
                        completed_at=None,
                    )
                )
            )
        budget_manager = BudgetManager(self._config)
        budget_state = budget_manager.start()
        normalized_query = self._normalize_query(query)
        tzinfo = resolve_timezone(self._config.time.timezone)
        now = dt.datetime.now(tzinfo)
        resolved_time_range = resolve_time_range_for_query(
            query=normalized_query,
            time_range=time_range,
            now=now,
            tzinfo=tzinfo,
        )
        time_only = is_time_only_expression(normalized_query)
        retrieve_query = "" if time_only else normalized_query
        retrieve_filters = None
        if filters:
            retrieve_filters = RetrieveFilters(
                apps=filters.get("app"), domains=filters.get("domain")
            )
        speculative = bool(
            self._config.retrieval.speculative_enabled
            and self._config.model_stages.draft_generate.enabled
        )
        draft_k = self._config.retrieval.speculative_draft_k if speculative else k
        final_k = self._config.retrieval.speculative_final_k if speculative else k
        retrieval_mode = "baseline" if speculative else None
        evidence_result = self._build_evidence(
            retrieve_query,
            resolved_time_range,
            filters,
            draft_k,
            sanitized,
            retrieval_mode=retrieval_mode,
            request_id=request_id,
        )
        evidence, events, no_evidence = _unpack_evidence_result(evidence_result)
        thread_aggregates = _build_thread_aggregates(
            self._thread_retrieval,
            normalized_query,
            resolved_time_range,
        )
        aggregates = _merge_aggregates(aggregates, thread_aggregates)
        if not evidence or no_evidence:
            warnings.append("no_evidence")
            empty_pack = build_context_pack(
                query=query,
                evidence=[],
                entity_tokens=[],
                routing=routing,
                filters={
                    "time_range": resolved_time_range,
                    "apps": filters.get("app") if filters else None,
                    "domains": filters.get("domain") if filters else None,
                },
                sanitized=sanitized,
                aggregates=aggregates,
            )
            payload = build_no_evidence_payload(
                query, has_time_range=resolved_time_range is not None
            )
            return _build_graph_result(
                payload["message"],
                [],
                empty_pack.to_json(),
                warnings,
                used_llm=False,
                output_format=output_format,
                context_pack_tron=None,
                prompt_strategy=None,
                mode="NO_EVIDENCE",
                hints=payload["hints"],
                actions=payload["actions"],
            )
        if (
            not time_only
            and self._should_refine(evidence)
            and self._config.model_stages.query_refine.enabled
            and not speculative
        ):
            base_evidence = evidence
            base_events = events
            base_no_evidence = no_evidence
            refined = await self._refine_query(
                normalized_query, evidence, routing_override=routing_override
            )
            refined_query = self._normalize_query(refined.refined_query or "")
            if len(refined_query) < 2:
                refined_query = ""
            if resolved_time_range is None:
                intent = resolve_time_intent(
                    time_expression=refined.time_expression,
                    time_range_payload=refined.time_range_payload,
                    now=now,
                    tzinfo=tzinfo,
                )
                resolved_time_range = intent.time_range
            if refined_query != normalized_query or (
                not refined_query and resolved_time_range is not None
            ):
                retrieve_query = refined_query
                refined_result = self._build_evidence(
                    refined_query,
                    resolved_time_range,
                    filters,
                    draft_k,
                    sanitized,
                    retrieval_mode=retrieval_mode,
                    request_id=request_id,
                )
                evidence, events, no_evidence = _unpack_evidence_result(refined_result)
                if no_evidence or not evidence:
                    warnings.append("no_evidence_refined")
                    evidence = base_evidence
                    events = base_events
                    no_evidence = base_no_evidence
        pack = build_context_pack(
            query=query,
            evidence=evidence,
            entity_tokens=self._entities.tokens_for_events(events),
            routing=routing,
            filters={
                "time_range": resolved_time_range,
                "apps": filters.get("app") if filters else None,
                "domains": filters.get("domain") if filters else None,
            },
            sanitized=sanitized,
            aggregates=aggregates,
        )
        context_pack_json = pack.to_json()
        pack_json_text = pack.to_text(extractive_only=False, format="json")
        pack_tron_text = (
            pack.to_text(extractive_only=False, format="tron")
            if context_pack_format == "tron"
            else None
        )
        context_pack_tron = pack_tron_text if context_pack_format == "tron" else None
        if time_only and resolved_time_range is not None:
            timeline = _build_timeline_answer(evidence)
            return self._finalize_answer(
                answer_text=timeline.answer,
                citations=timeline.citations,
                context_pack_json=context_pack_json,
                warnings=warnings,
                used_llm=False,
                output_format=output_format,
                context_pack_tron=context_pack_tron,
                prompt_strategy_payload=None,
                query=query,
                retrieve_filters=retrieve_filters,
                resolved_time_range=resolved_time_range,
                budget_manager=budget_manager,
                budget_state=budget_state,
                evidence=evidence,
            )
        if extractive_only:
            compressor = self._get_compressor()
            compressed = compressor.compress(evidence)
            return self._finalize_answer(
                answer_text=compressed.answer,
                citations=compressed.citations,
                context_pack_json=context_pack_json,
                warnings=warnings,
                used_llm=False,
                output_format=output_format,
                context_pack_tron=context_pack_tron,
                prompt_strategy_payload=None,
                query=query,
                retrieve_filters=retrieve_filters,
                resolved_time_range=resolved_time_range,
                budget_manager=budget_manager,
                budget_state=budget_state,
                evidence=evidence,
            )

        system_prompt = self._prompt_registry.get("ANSWER_WITH_CONTEXT_PACK").system_prompt
        draft_text: str | None = None
        draft_citations: list[str] = []
        draft_valid = False
        draft_claims_payload = None
        provider_calls: list[dict] = []
        speculative_confident = False
        if speculative:
            score, gap = self._evidence_confidence(evidence)
            speculative_confident = _is_confident(score, gap, self._config.retrieval)
        if self._config.model_stages.draft_generate.enabled:
            draft_decision = None
            try:
                draft_provider, draft_decision = self._stage_router.select_llm(
                    "draft_generate", routing_override=routing_override
                )
                draft_query = _build_draft_query(query)
                draft_pack = _select_context_pack_text(
                    self._config,
                    draft_decision,
                    context_pack_format,
                    pack_json_text,
                    pack_tron_text,
                    warnings,
                    stage="draft_generate",
                )
                draft_start = time.monotonic()
                with otel_span("answergraph.stage", {"stage_name": "draft_generate"}):
                    with otel_span(
                        "llm.call",
                        {
                            "stage_name": "draft_generate",
                            "provider_id": getattr(draft_decision, "provider", "unknown"),
                            "model_id": getattr(draft_decision, "model_id", None),
                        },
                    ):
                        draft_text = await draft_provider.generate_answer(
                            system_prompt,
                            draft_query,
                            draft_pack,
                            temperature=draft_decision.temperature,
                        )
                record_histogram(
                    "answer_generate_ms",
                    (time.monotonic() - draft_start) * 1000,
                    {"stage_name": "answer_generate"},
                )
                provider_calls.append(
                    {
                        "stage": "draft_generate",
                        "provider_id": getattr(draft_decision, "provider", "unknown"),
                        "model_id": getattr(draft_decision, "model_id", None),
                        "attempt_index": 1,
                        "success": True,
                        "status_code": None,
                        "error_text": None,
                        "latency_ms": (time.monotonic() - draft_start) * 1000,
                    }
                )
                budget_manager.record_stage(
                    budget_state, "answer_draft", (time.monotonic() - draft_start) * 1000
                )
                verifier = self._get_verifier()
                if self._requires_claims(draft_decision):
                    (
                        draft_text,
                        draft_citations,
                        draft_valid,
                        draft_claims_payload,
                        draft_errors,
                    ) = self._process_claims_output(draft_text, evidence, verifier)
                    if draft_errors:
                        warnings.append("draft_claims_invalid")
                        verification_failures_total.labels("draft", "claims_invalid").inc()
                else:
                    draft_citations = _extract_citations(draft_text)
                    draft_valid = _verify_answer(draft_text, draft_citations, evidence, verifier)
            except Exception as exc:
                warnings.append("draft_failed")
                provider_calls.append(
                    {
                        "stage": "draft_generate",
                        "provider_id": getattr(draft_decision, "provider", "unknown"),
                        "model_id": getattr(draft_decision, "model_id", None),
                        "attempt_index": 1,
                        "success": False,
                        "status_code": None,
                        "error_text": str(exc),
                        "latency_ms": None,
                    }
                )
                self._log.warning("Draft generation failed; skipping: {}", exc)

        if speculative and draft_valid and draft_text is not None and speculative_confident:
            if self._requires_claims(draft_decision) and draft_claims_payload:
                return self._finalize_answer(
                    answer_text=draft_text,
                    citations=draft_citations,
                    context_pack_json=context_pack_json,
                    warnings=warnings,
                    used_llm=True,
                    output_format=output_format,
                    context_pack_tron=context_pack_tron,
                    prompt_strategy_payload=None,
                    query=query,
                    retrieve_filters=retrieve_filters,
                    resolved_time_range=resolved_time_range,
                    budget_manager=budget_manager,
                    budget_state=budget_state,
                    evidence=evidence,
                    claims_payload=draft_claims_payload,
                )
            if not self._config.verification.claims_enabled:
                return self._finalize_answer(
                    answer_text=draft_text,
                    citations=draft_citations,
                    context_pack_json=context_pack_json,
                    warnings=warnings,
                    used_llm=True,
                    output_format=output_format,
                    context_pack_tron=context_pack_tron,
                    prompt_strategy_payload=None,
                    query=query,
                    retrieve_filters=retrieve_filters,
                    resolved_time_range=resolved_time_range,
                    budget_manager=budget_manager,
                    budget_state=budget_state,
                    evidence=evidence,
                )

        if speculative:
            deep_result = self._build_evidence(
                retrieve_query,
                resolved_time_range,
                filters,
                final_k,
                sanitized,
                retrieval_mode="deep",
                request_id=request_id,
            )
            evidence, events, no_evidence = _unpack_evidence_result(deep_result)
            if not evidence or no_evidence:
                warnings.append("no_evidence_deep")
            pack = build_context_pack(
                query=query,
                evidence=evidence,
                entity_tokens=self._entities.tokens_for_events(events),
                routing=routing,
                filters={
                    "time_range": resolved_time_range,
                    "apps": filters.get("app") if filters else None,
                    "domains": filters.get("domain") if filters else None,
                },
                sanitized=sanitized,
                aggregates=aggregates,
            )
            context_pack_json = pack.to_json()
            pack_json_text = pack.to_text(extractive_only=False, format="json")
            pack_tron_text = (
                pack.to_text(extractive_only=False, format="tron")
                if context_pack_format == "tron"
                else None
            )
            context_pack_tron = pack_tron_text if context_pack_format == "tron" else None

        answer_text = ""
        citations: list[str] = []
        used_llm = False
        final_valid = False
        prompt_strategy_payload: dict | None = None
        final_claims_payload = None
        final_decision = None
        final_errors: list[str] = []
        try:
            final_provider, final_decision = self._stage_router.select_llm(
                "final_answer", routing_override=routing_override
            )
            final_pack = _select_context_pack_text(
                self._config,
                final_decision,
                context_pack_format,
                pack_json_text,
                pack_tron_text,
                warnings,
                stage="final_answer",
            )
            base_final_query = _build_final_query(query, draft_text)
            max_attempts = _stage_max_attempts(final_decision)
            attempt = 0
            verifier = self._get_verifier()
            while attempt < max_attempts:
                attempt += 1
                final_query = base_final_query
                if self._requires_claims(final_decision):
                    final_query = self._append_claims_instructions(
                        base_final_query, final_errors if attempt > 1 else None
                    )
                answer_start = time.monotonic()
                with otel_span("answergraph.stage", {"stage_name": "final_answer"}):
                    with otel_span(
                        "llm.call",
                        {
                            "stage_name": "final_answer",
                            "provider_id": getattr(final_decision, "provider", "unknown"),
                            "model_id": getattr(final_decision, "model_id", None),
                        },
                    ):
                        answer_text = await final_provider.generate_answer(
                            system_prompt,
                            final_query,
                            final_pack,
                            temperature=0.0 if attempt > 1 else final_decision.temperature,
                        )
                latency_ms = (time.monotonic() - answer_start) * 1000
                record_histogram(
                    "answer_generate_ms",
                    latency_ms,
                    {"stage_name": "answer_generate"},
                )
                provider_calls.append(
                    {
                        "stage": "final_answer",
                        "provider_id": getattr(final_decision, "provider", "unknown"),
                        "model_id": getattr(final_decision, "model_id", None),
                        "attempt_index": attempt,
                        "success": True,
                        "status_code": None,
                        "error_text": None,
                        "latency_ms": latency_ms,
                    }
                )
                budget_manager.record_stage(budget_state, "answer_final", latency_ms)
                prompt_strategy_payload = _prompt_strategy_payload(final_provider)
                if self._requires_claims(final_decision):
                    (
                        answer_text,
                        citations,
                        final_valid,
                        final_claims_payload,
                        final_errors,
                    ) = self._process_claims_output(answer_text, evidence, verifier)
                    if final_errors:
                        warnings.append("final_claims_invalid")
                        verification_failures_total.labels("final", "claims_invalid").inc()
                    if final_valid:
                        break
                else:
                    citations = _extract_citations(answer_text)
                    final_valid = _verify_answer(answer_text, citations, evidence, verifier)
                    if final_valid:
                        break
            if (
                not final_valid
                and self._requires_claims(final_decision)
                and getattr(final_decision, "policy", None)
                and final_decision.policy.repair_on_failure
            ):
                regen = await self._regenerate_with_deep_retrieval(
                    query=query,
                    draft_text=draft_text,
                    retrieve_query=retrieve_query,
                    resolved_time_range=resolved_time_range,
                    filters=filters,
                    sanitized=sanitized,
                    aggregates=aggregates,
                    final_k=final_k,
                    routing=routing,
                    routing_override=routing_override,
                    context_pack_format=context_pack_format,
                    system_prompt=system_prompt,
                    budget_manager=budget_manager,
                    budget_state=budget_state,
                    request_id=request_id,
                )
                if regen is not None:
                    (
                        answer_text,
                        citations,
                        final_claims_payload,
                        evidence,
                        context_pack_json,
                        context_pack_tron,
                    ) = regen
                    final_valid = True
        except Exception as exc:
            warnings.append("final_failed")
            provider_calls.append(
                {
                    "stage": "final_answer",
                    "provider_id": getattr(final_decision, "provider", "unknown"),
                    "model_id": getattr(final_decision, "model_id", None),
                    "attempt_index": 1,
                    "success": False,
                    "status_code": None,
                    "error_text": str(exc),
                    "latency_ms": None,
                }
            )
            self._log.warning("Final answer failed; falling back: {}", exc)

        selected_claims_payload = None
        claims_required = bool(final_decision and self._requires_claims(final_decision))
        if final_valid and (not draft_valid or len(citations) >= len(draft_citations)):
            used_llm = True
            selected_claims_payload = final_claims_payload
        elif (
            draft_valid and draft_text is not None and (not claims_required or draft_claims_payload)
        ):
            answer_text = draft_text
            citations = draft_citations
            used_llm = True
            selected_claims_payload = draft_claims_payload
        elif claims_required:
            warnings.append("claims_validation_blocked")
            answer_text = "Answer withheld: citation validation failed."
            citations = []
            used_llm = False
        else:
            compressed = extractive_answer(evidence)
            answer_text = compressed.answer
            citations = compressed.citations
            used_llm = False

        if selected_claims_payload and self._config.verification.entailment.enabled:
            entailment = await self._apply_entailment_gate(
                selected_claims_payload,
                evidence,
                warnings,
                query=query,
                draft_text=draft_text,
                retrieve_query=retrieve_query,
                resolved_time_range=resolved_time_range,
                filters=filters,
                sanitized=sanitized,
                aggregates=aggregates,
                final_k=final_k,
                routing=routing,
                routing_override=routing_override,
                context_pack_format=context_pack_format,
                context_pack_json=context_pack_json,
                context_pack_tron=context_pack_tron,
                system_prompt=system_prompt,
                budget_manager=budget_manager,
                budget_state=budget_state,
                request_id=request_id,
            )
            if entailment is not None:
                (
                    answer_text,
                    citations,
                    selected_claims_payload,
                    evidence,
                    context_pack_json,
                    context_pack_tron,
                ) = entailment
        return self._finalize_answer(
            answer_text=answer_text,
            citations=citations,
            context_pack_json=context_pack_json,
            warnings=warnings,
            used_llm=used_llm,
            output_format=output_format,
            context_pack_tron=context_pack_tron,
            prompt_strategy_payload=prompt_strategy_payload,
            query=query,
            retrieve_filters=retrieve_filters,
            resolved_time_range=resolved_time_range,
            budget_manager=budget_manager,
            budget_state=budget_state,
            evidence=evidence,
            claims_payload=selected_claims_payload,
            provider_calls=provider_calls,
            request_id=request_id,
            request_started_at=request_started_at,
        )

    def _finalize_answer(
        self,
        *,
        answer_text: str,
        citations: list[str],
        context_pack_json: dict,
        warnings: list[str],
        used_llm: bool,
        output_format: str,
        context_pack_tron: str | None,
        prompt_strategy_payload: dict | None,
        query: str,
        retrieve_filters: RetrieveFilters | None,
        resolved_time_range: tuple[dt.datetime, dt.datetime] | None,
        budget_manager: BudgetManager,
        budget_state,
        evidence: list[EvidenceItem],
        claims_payload: object | None = None,
        provider_calls: list[dict] | None = None,
        request_id: str | None = None,
        request_started_at: dt.datetime | None = None,
    ) -> AnswerGraphResult:
        query_class = _classify_query(query, retrieve_filters)
        query_id = _ensure_query_record(
            self._retrieval._db,  # type: ignore[attr-defined]
            query,
            retrieve_filters,
            query_class,
            budget_manager.snapshot(),
        )
        evidence_by_id = {item.evidence_id: item for item in evidence}
        span_ids: list[str] = []
        for item in evidence_by_id.values():
            for span in item.spans:
                if span.span_id:
                    span_ids.append(str(span.span_id))
        has_citable_spans = False
        if hasattr(self._retrieval, "_db"):
            event_ids = [item.event_id for item in evidence_by_id.values()]
            if event_ids:
                with self._retrieval._db.session() as session:  # type: ignore[attr-defined]
                    has_citable_spans = (
                        session.execute(
                            select(CitableSpanRecord.span_id).where(
                                CitableSpanRecord.frame_id.in_(event_ids)
                            )
                        )
                        .scalars()
                        .first()
                        is not None
                    )
        if has_citable_spans:
            integrity = check_citations(self._retrieval._db, span_ids)  # type: ignore[attr-defined]
            valid_span_ids = integrity.valid_span_ids
            if self._config.next10.enabled:
                provenance = verify_provenance(
                    self._retrieval._db,  # type: ignore[attr-defined]
                    query_id=query_id,
                    span_ids=list(valid_span_ids),
                )
                if provenance.missing:
                    warnings.append("provenance_missing")
                valid_span_ids = provenance.valid_span_ids
            valid_evidence_ids = {
                evidence_id
                for evidence_id, item in evidence_by_id.items()
                if any(span.span_id and str(span.span_id) in valid_span_ids for span in item.spans)
            }
            filtered_citations = [c for c in citations if c in valid_evidence_ids]
            if set(filtered_citations) != set(citations):
                warnings.append("citation_integrity_failed")
        else:
            integrity = check_citations(self._retrieval._db, [])  # type: ignore[attr-defined]
            valid_span_ids = set()
            valid_evidence_ids = set(evidence_by_id)
            filtered_citations = list(citations)

        coverage = coverage_metrics(
            answer_text,
            valid_evidence_ids,
            no_evidence_mode=False,
        )
        coverage["evidence_count"] = len(valid_span_ids)
        thresholds = _coverage_thresholds(self._config)
        blocked = any(
            flag in warnings
            for flag in (
                "claims_validation_blocked",
                "entailment_contradicted",
            )
        )
        confidence = {
            "coverage": coverage.get("sentence_coverage", 0.0),
            "integrity_ok": not bool(integrity.invalid_span_ids),
            "citation_count": len(filtered_citations),
        }
        mode = "NORMAL"
        conflict = detect_conflicts(evidence)
        conflict_summary = (
            conflict.summary if conflict.conflict or conflict.changed_over_time else None
        )
        if conflict.conflict:
            mode = "CONFLICT"
            answer_text = _conflict_message(conflict.summary)
        if mode != "CONFLICT" and blocked:
            mode = "BLOCKED"
        if mode not in {"CONFLICT", "BLOCKED"}:
            if not valid_evidence_ids or coverage.get("sentence_coverage", 0.0) < thresholds[1]:
                mode = "NO_EVIDENCE"
            elif coverage.get("sentence_coverage", 0.0) < thresholds[0]:
                mode = "NO_EVIDENCE"

        hints: list[dict] | None = None
        actions: list[dict] | None = None
        if mode == "NO_EVIDENCE":
            payload = build_no_evidence_payload(
                query, has_time_range=resolved_time_range is not None
            )
            answer_text = payload["message"]
            citations = []
            hints = payload["hints"]
            actions = payload["actions"]
            coverage = coverage_metrics(
                answer_text,
                valid_evidence_ids,
                no_evidence_mode=True,
            )
            coverage["evidence_count"] = len(valid_span_ids)
            confidence["coverage"] = coverage.get("sentence_coverage", 0.0)
        else:
            citations = filtered_citations
            hints = []
            actions = []

        answer_id = stable_id(
            "answer",
            {"query_id": query_id, "answer_text": answer_text, "citations": citations},
        )
        sentence_citations = extract_sentence_citations(answer_text, valid_evidence_ids)
        evidence_to_span = {
            evidence_id: next(
                (
                    str(span.span_id)
                    for span in item.spans
                    if span.span_id and str(span.span_id) in valid_span_ids
                ),
                None,
            )
            for evidence_id, item in evidence_by_id.items()
        }
        line_maps: list[dict] = []
        for evidence_id, item in evidence_by_id.items():
            line_text = item.text or item.raw_text or ""
            offsets = _line_offsets(line_text)
            line_maps.append(
                {
                    "map_id": stable_id(
                        "line_map",
                        {
                            "query_id": query_id,
                            "evidence_id": evidence_id,
                            "text_sha256": sha256_text(line_text),
                        },
                    ),
                    "query_id": query_id,
                    "evidence_id": evidence_id,
                    "span_id": evidence_to_span.get(evidence_id),
                    "line_count": len(offsets),
                    "line_offsets": offsets,
                    "text_sha256": sha256_text(line_text),
                }
            )
        budgets_json = {
            "stage_ms_used": budget_state.stage_ms_used,
            "degraded_stages": budget_state.degraded_stages,
        }

        def _persist_answer(session) -> None:
            now = dt.datetime.now(dt.timezone.utc)
            existing = session.get(AnswerRecord, answer_id)
            if not existing:
                session.add(
                    AnswerRecord(
                        answer_id=answer_id,
                        query_id=query_id,
                        mode=mode,
                        coverage_json=coverage,
                        confidence_json=confidence,
                        budgets_json=budgets_json,
                        answer_text=answer_text,
                        stale=bool(integrity.invalid_span_ids),
                        answer_format_version=1,
                        schema_version=1,
                        created_at=now,
                    )
                )
                session.flush()
            if request_id:
                run = session.get(RequestRunRecord, request_id)
                if run:
                    run.query_id = query_id
                    run.status = "completed"
                    run.warnings_json = {"warnings": warnings}
                    run.completed_at = now
            existing_pairs: set[tuple[str, str | None]] = set()
            if existing:
                rows = session.execute(
                    select(AnswerCitationRecord.sentence_id, AnswerCitationRecord.span_id).where(
                        AnswerCitationRecord.answer_id == answer_id
                    )
                ).all()
                existing_pairs = {(row[0], row[1]) for row in rows}
            for sentence in sentence_citations:
                for citation in sentence.citations:
                    span_id = evidence_to_span.get(citation)
                    if span_id is None:
                        continue
                    if (sentence.sentence_id, span_id) in existing_pairs:
                        continue
                    session.add(
                        AnswerCitationRecord(
                            answer_id=answer_id,
                            sentence_id=sentence.sentence_id,
                            sentence_index=sentence.index,
                            span_id=span_id,
                            citable=True,
                            created_at=now,
                        )
                    )
            if claims_payload and getattr(claims_payload, "claims", None):
                for idx, claim in enumerate(claims_payload.claims, start=1):
                    claim_id = claim.claim_id or stable_id(
                        "claim",
                        {"answer_id": answer_id, "index": idx, "text": claim.text},
                    )
                    verdict = None
                    if getattr(claims_payload, "entailment", None):
                        verdict = claims_payload.entailment.get(claim_id)
                    session.add(
                        AnswerClaimRecord(
                            claim_id=claim_id,
                            answer_id=answer_id,
                            claim_index=idx,
                            claim_text=claim.text,
                            entailment_verdict=verdict,
                            entailment_rationale=None,
                            schema_version=1,
                            created_at=now,
                        )
                    )
                    with session.no_autoflush:
                        existing_claim = session.get(ClaimRecord, claim_id)
                    if not existing_claim:
                        session.add(
                            ClaimRecord(
                                claim_id=claim_id,
                                request_id=request_id,
                                claim_text=claim.text,
                                entailment_verdict=verdict,
                                created_at=now,
                            )
                        )
                    session.flush()
                    citations = list(getattr(claim, "citations", []) or [])
                    if not citations:
                        citations = [
                            SimpleNamespace(
                                evidence_id=eid,
                                line_start=1,
                                line_end=1,
                                confidence=None,
                            )
                            for eid in claim.evidence_ids
                        ]
                    for cite in citations:
                        evidence_id = getattr(cite, "evidence_id", None) or ""
                        span_id = evidence_to_span.get(evidence_id)
                        session.add(
                            AnswerClaimCitationRecord(
                                claim_id=claim_id,
                                span_id=span_id,
                                evidence_id=evidence_id,
                                line_start=getattr(cite, "line_start", None),
                                line_end=getattr(cite, "line_end", None),
                                confidence=getattr(cite, "confidence", None),
                                created_at=now,
                            )
                        )
                        if evidence_id:
                            line_start = getattr(cite, "line_start", None) or 1
                            line_end = getattr(cite, "line_end", None) or line_start
                            session.add(
                                ClaimCitationRecord(
                                    claim_id=claim_id,
                                    evidence_id=evidence_id,
                                    line_start=int(line_start),
                                    line_end=int(line_end),
                                    confidence=getattr(cite, "confidence", None),
                                    created_at=now,
                                )
                            )
            for line_map in line_maps:
                existing = session.get(EvidenceLineMapRecord, line_map["map_id"])
                if existing is None:
                    session.add(
                        EvidenceLineMapRecord(
                            map_id=line_map["map_id"],
                            query_id=line_map["query_id"],
                            evidence_id=line_map["evidence_id"],
                            span_id=line_map["span_id"],
                            line_count=line_map["line_count"],
                            line_offsets_json=line_map["line_offsets"],
                            text_sha256=line_map["text_sha256"],
                            created_at=now,
                        )
                    )
            if request_id:
                line_count_map = {item["evidence_id"]: item["line_count"] for item in line_maps}
                for evidence_id, item in evidence_by_id.items():
                    item_id = stable_id(
                        "evidence_item",
                        {
                            "request_id": request_id,
                            "evidence_id": evidence_id,
                            "content_hash": item.content_hash,
                        },
                    )
                    if session.get(EvidenceItemRecord, item_id):
                        continue
                    session.add(
                        EvidenceItemRecord(
                            item_id=item_id,
                            request_id=request_id,
                            query_id=query_id,
                            evidence_id=evidence_id,
                            event_id=item.event_id,
                            content_hash=item.content_hash or sha256_text(item.text or ""),
                            line_count=line_count_map.get(evidence_id, 0),
                            injection_risk=item.injection_risk,
                            citable=item.citable,
                            kind=item.kind,
                            created_at=now,
                        )
                    )
            if provider_calls:
                for idx, call in enumerate(provider_calls, start=1):
                    call_id = stable_id(
                        "provider_call",
                        {
                            "answer_id": answer_id,
                            "query_id": query_id,
                            "stage": call.get("stage"),
                            "provider_id": call.get("provider_id"),
                            "attempt_index": call.get("attempt_index", idx),
                            "success": call.get("success"),
                        },
                    )
                    if session.get(ProviderCallRecord, call_id):
                        continue
                    session.add(
                        ProviderCallRecord(
                            call_id=call_id,
                            query_id=query_id,
                            answer_id=answer_id,
                            stage=str(call.get("stage") or ""),
                            provider_id=str(call.get("provider_id") or ""),
                            model_id=call.get("model_id"),
                            attempt_index=int(call.get("attempt_index") or idx),
                            success=bool(call.get("success")),
                            status_code=call.get("status_code"),
                            error_text=call.get("error_text"),
                            latency_ms=call.get("latency_ms"),
                            created_at=now,
                        )
                    )
                    if request_id:
                        stage_run_id = stable_id(
                            "stage_run",
                            {
                                "request_id": request_id,
                                "stage": call.get("stage"),
                                "provider_id": call.get("provider_id"),
                                "attempt_index": call.get("attempt_index", idx),
                                "success": call.get("success"),
                                "ts": now.isoformat(),
                            },
                        )
                        if not session.get(StageRunRecord, stage_run_id):
                            session.add(
                                StageRunRecord(
                                    run_id=stage_run_id,
                                    request_id=request_id,
                                    stage=str(call.get("stage") or ""),
                                    provider_id=str(call.get("provider_id") or ""),
                                    model_id=call.get("model_id"),
                                    attempt_index=int(call.get("attempt_index") or idx),
                                    success=bool(call.get("success")),
                                    status_code=call.get("status_code"),
                                    error_text=call.get("error_text"),
                                    latency_ms=call.get("latency_ms"),
                                    created_at=now,
                                )
                            )
                        provider_id = str(call.get("provider_id") or "")
                        if provider_id:
                            health = session.get(ProviderHealthRecord, provider_id)
                            if not health:
                                health = ProviderHealthRecord(
                                    provider_id=provider_id,
                                    consecutive_failures=0,
                                    circuit_open_until=None,
                                    last_error=None,
                                    updated_at=now,
                                )
                                session.add(health)
                            if call.get("success"):
                                health.consecutive_failures = 0
                                health.last_error = None
                            else:
                                health.consecutive_failures = (
                                    int(health.consecutive_failures or 0) + 1
                                )
                                health.last_error = call.get("error_text")
                            health.updated_at = now

        self._retrieval._db.transaction(_persist_answer)  # type: ignore[attr-defined]
        update_tier_stats(
            self._retrieval._db,  # type: ignore[attr-defined]
            query_id=query_id,
            query_class=query_class,
            cited_span_ids=valid_span_ids,
        )
        sentence_payload = [
            {"sentence_id": s.sentence_id, "citations": s.citations} for s in sentence_citations
        ]
        chain_appended = False
        if self._config.next10.enabled:
            chain_appended = append_provenance_chain(
                self._config,
                self._retrieval._db,  # type: ignore[attr-defined]
                self._ledger,
                answer_id=answer_id,
                query_id=query_id,
                evidence_to_span=evidence_to_span,
                sentence_citations=sentence_payload,
            )
        if chain_appended:
            try:
                self._ledger.append_entry(
                    "answer",
                    {
                        "answer_id": answer_id,
                        "query_id": query_id,
                        "mode": mode,
                        "citations": citations,
                        "span_ids": sorted(valid_span_ids),
                        "coverage": coverage,
                        "confidence": confidence,
                    },
                    answer_id=answer_id,
                )
            except Exception as exc:  # pragma: no cover - best effort
                self._log.debug("Failed to append answer ledger entry: {}", exc)

        return _build_graph_result(
            answer_text,
            citations,
            context_pack_json,
            warnings,
            used_llm=used_llm,
            output_format=output_format,
            context_pack_tron=context_pack_tron,
            prompt_strategy=prompt_strategy_payload,
            mode=mode,
            coverage=coverage,
            confidence=confidence,
            budgets=budgets_json,
            degraded_stages=budget_state.degraded_stages,
            hints=hints,
            actions=actions,
            conflict_summary=conflict_summary,
            answer_id=answer_id,
            query_id=query_id,
            claims=(
                claims_payload.model_dump(mode="json", by_alias=True)
                if hasattr(claims_payload, "model_dump")
                else None
            ),
        )

    def _build_evidence(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: dict[str, list[str]] | None,
        k: int,
        sanitized: bool,
        retrieval_mode: str | None = None,
        request_id: str | None = None,
    ) -> tuple[list[EvidenceItem], list[EventRecord], bool]:
        retrieve_filters = None
        if filters:
            retrieve_filters = RetrieveFilters(
                apps=filters.get("app"), domains=filters.get("domain")
            )
        if self._config.next10.enabled and hasattr(self._retrieval, "retrieve_tiered"):
            batch = self._retrieval.retrieve_tiered(
                query,
                time_range,
                retrieve_filters,
                limit=k,
                mode=retrieval_mode,
                request_id=request_id,
            )
        else:
            batch = self._retrieval.retrieve(
                query,
                time_range,
                retrieve_filters,
                limit=k,
                mode=retrieval_mode,
                request_id=request_id,
            )
        results = list(batch.results)
        no_evidence = bool(batch.no_evidence)
        if not results and time_range and not query:
            batch = self._retrieval.retrieve("", time_range, retrieve_filters, limit=k)
            results = list(batch.results)
            no_evidence = bool(batch.no_evidence)
        if not results:
            return [], [], True
        span_lookup: dict[tuple[str, str], CitableSpanRecord] = {}
        fallback_spans: dict[str, CitableSpanRecord] = {}
        if hasattr(self._retrieval, "_db"):
            event_ids = [item.event.event_id for item in results]
            if event_ids:
                with self._retrieval._db.session() as session:  # type: ignore[attr-defined]
                    rows = (
                        session.execute(
                            select(CitableSpanRecord).where(
                                CitableSpanRecord.frame_id.in_(event_ids)
                            )
                        )
                        .scalars()
                        .all()
                    )
                for row in rows:
                    if row.frame_id not in fallback_spans:
                        fallback_spans[row.frame_id] = row
                    if row.legacy_span_key:
                        span_lookup[(row.frame_id, row.legacy_span_key)] = row
        raw_items: list[dict[str, Any]] = []
        events: list[EventRecord] = []
        seen_keys: set[tuple[str, str]] = set()
        for result in results:
            event = result.event
            snippet = result.snippet or (event.ocr_text or "")[:500]
            raw_snippet = snippet
            frame_size = _frame_size_from_tags(event.tags)
            spans: list[EvidenceSpan] = []
            matched_span = None
            if result.matched_span_keys:
                for key in result.matched_span_keys:
                    matched_span = span_lookup.get((event.event_id, str(key)))
                    if matched_span:
                        break
            if matched_span is None:
                matched_span = fallback_spans.get(event.event_id)
            bbox = result.bbox
            bbox_norm = None
            span_id = result.matched_span_keys[0] if result.matched_span_keys else "S0"
            if matched_span:
                span_id = matched_span.span_id
                if matched_span.bbox is not None:
                    bbox = matched_span.bbox
                if matched_span.bbox_norm is not None:
                    bbox_norm = matched_span.bbox_norm
            if bbox and bbox_norm is None:
                bbox_norm = _bbox_norm(bbox, frame_size)
            if bbox:
                spans.append(
                    EvidenceSpan(
                        span_id=str(span_id),
                        start=0,
                        end=len(snippet),
                        conf=0.5,
                        bbox=bbox,
                        bbox_norm=bbox_norm,
                    )
                )
            app_name = event.app_name
            title = event.window_title
            domain = event.domain
            if sanitized:
                snippet = self._entities.pseudonymize_text(snippet)
                app_name = self._entities.pseudonymize_text(app_name)
                title = self._entities.pseudonymize_text(title)
                if domain:
                    domain = self._entities.pseudonymize_text(domain)
            scan = scan_prompt_injection(snippet)
            redacted_text = scan.redacted_text
            content_hash = sha256_text(redacted_text)
            key = (event.event_id, content_hash)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            events.append(event)
            raw_items.append(
                {
                    "key": key,
                    "event": event,
                    "app_name": app_name,
                    "title": title,
                    "domain": domain,
                    "score": result.score,
                    "spans": spans,
                    "text": redacted_text,
                    "raw_text": raw_snippet,
                    "redacted_text": redacted_text if scan.match_count else None,
                    "injection_risk": scan.risk_score,
                    "content_hash": content_hash,
                    "non_citable": getattr(result, "non_citable", False),
                    "retrieval": {
                        "engine": getattr(result, "engine", "hybrid"),
                        "rank": getattr(result, "rank", 0),
                        "rank_gap": getattr(result, "rank_gap", 0.0),
                        "lexical_score": getattr(result, "lexical_score", 0.0),
                        "vector_score": getattr(result, "vector_score", 0.0),
                        "sparse_score": getattr(result, "sparse_score", 0.0),
                        "late_score": getattr(result, "late_score", 0.0),
                        "rerank_score": getattr(result, "rerank_score", None),
                        "matched_spans": getattr(result, "matched_span_keys", []),
                        "ts_start": event.ts_start.isoformat(),
                        "non_citable": getattr(result, "non_citable", False),
                        "query_id": getattr(batch, "query_id", None),
                        "tier_plan": getattr(batch, "tier_plan", None),
                    },
                }
            )
        if not raw_items:
            return [], [], True
        sorted_keys = sorted({item["key"] for item in raw_items})
        key_to_id = {key: f"E{idx}" for idx, key in enumerate(sorted_keys, start=1)}
        evidence: list[EvidenceItem] = []
        for item in raw_items:
            evidence_id = key_to_id[item["key"]]
            non_citable = bool(item["non_citable"])
            evidence.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    event_id=item["event"].event_id,
                    timestamp=item["event"].ts_start.isoformat(),
                    ts_end=item["event"].ts_end.isoformat() if item["event"].ts_end else None,
                    app=item["app_name"],
                    title=item["title"],
                    domain=item["domain"],
                    score=item["score"],
                    spans=item["spans"],
                    text=item["text"],
                    raw_text=item["raw_text"],
                    redacted_text=item["redacted_text"],
                    kind="derived_summary" if non_citable else "source",
                    citable=not non_citable,
                    injection_risk=item["injection_risk"],
                    content_hash=item["content_hash"],
                    screenshot_path=item["event"].screenshot_path,
                    screenshot_hash=item["event"].screenshot_hash,
                    retrieval=item["retrieval"],
                )
            )
        return evidence, events, no_evidence

    def _normalize_query(self, query: str) -> str:
        return query.strip()

    def _should_refine(self, evidence: list) -> bool:
        return len(evidence) < 3

    async def _refine_query(
        self, query: str, evidence: list, *, routing_override: str | None
    ) -> RefinedQueryIntent:
        fallback_query = self._heuristic_refine_query(query, evidence)
        fallback = RefinedQueryIntent(
            refined_query=fallback_query,
            time_expression=None,
            time_range_payload=None,
        )
        try:
            prompt = self._prompt_registry.get("QUERY_REFINEMENT")
        except Exception as exc:
            self._log.warning("Query refinement prompt unavailable: {}", exc)
            return fallback
        try:
            provider, decision = self._stage_router.select_llm(
                "query_refine", routing_override=routing_override
            )
            context = self._build_refinement_context(evidence)
            refine_start = time.monotonic()
            with otel_span("answer_generate", {"stage_name": "answer_generate"}):
                response = await provider.generate_answer(
                    prompt.system_prompt,
                    query,
                    context,
                    temperature=decision.temperature,
                )
            record_histogram(
                "answer_generate_ms",
                (time.monotonic() - refine_start) * 1000,
                {"stage_name": "answer_generate"},
            )
            refined = _parse_refined_query(response)
            if refined:
                refined_query = refined.refined_query or fallback_query
                return RefinedQueryIntent(
                    refined_query=refined_query,
                    time_expression=refined.time_expression,
                    time_range_payload=refined.time_range_payload,
                )
        except Exception as exc:
            self._log.warning("Query refinement failed; using fallback: {}", exc)
        return fallback

    def _heuristic_refine_query(self, query: str, evidence: list) -> str:
        tokens = []
        for item in evidence:
            app = getattr(item, "app", None)
            if app:
                tokens.append(app)
        if not tokens:
            return query
        return f"{query} {' '.join(tokens[:3])}"

    def _build_refinement_context(self, evidence: list) -> str:
        apps = _dedupe([getattr(item, "app", "") for item in evidence])
        titles = _dedupe([getattr(item, "title", "") for item in evidence])
        domains = _dedupe([getattr(item, "domain", "") for item in evidence])
        snippets = []
        for item in evidence[:5]:
            text = getattr(item, "text", "") or ""
            text = " ".join(text.split())
            if text:
                snippets.append(text[:160])
        payload = {
            "apps": apps[:5],
            "titles": titles[:5],
            "domains": domains[:5],
            "snippets": snippets,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _merge_routing(self, routing: dict[str, str]) -> Any:
        from ..config import ProviderRoutingConfig

        return ProviderRoutingConfig(**routing)

    def _evidence_confidence(self, evidence: list[EvidenceItem]) -> tuple[float, float]:
        scores = [item.score for item in evidence if isinstance(item.score, (int, float))]
        if not scores:
            return 0.0, 0.0
        scores.sort(reverse=True)
        top = scores[0]
        second = scores[1] if len(scores) > 1 else 0.0
        return top, max(0.0, top - second)

    def _get_compressor(self):
        if self._compressor is not None:
            return self._compressor
        compressor_id = (self._config.routing.compressor or "extractive").strip().lower()
        try:
            self._compressor = self._plugins.resolve_extension(
                "compressor",
                compressor_id,
            )
        except Exception as exc:
            self._log.warning("Compressor plugin failed ({}): {}", compressor_id, exc)
            self._compressor = None
        if self._compressor is None:
            self._compressor = _ExtractiveWrapper()
        return self._compressor

    def _get_verifier(self):
        if self._verifier is not None:
            return self._verifier
        verifier_id = (self._config.routing.verifier or "rules").strip().lower()
        try:
            self._verifier = self._plugins.resolve_extension("verifier", verifier_id)
        except Exception as exc:
            self._log.warning("Verifier plugin failed ({}): {}", verifier_id, exc)
            self._verifier = RulesVerifier()
        return self._verifier

    def _requires_claims(self, decision: object) -> bool:
        requirements = getattr(decision, "requirements", None)
        if not requirements:
            if not self._config.verification.claims_enabled:
                return False
            return getattr(decision, "stage", "") == "final_answer"
        if bool(requirements.require_json and requirements.claims_schema == "claims_json_v1"):
            return True
        return bool(
            self._config.verification.claims_enabled
            and getattr(decision, "stage", "") == "final_answer"
        )

    def _append_claims_instructions(self, query: str, errors: list[str] | None = None) -> str:
        instructions = (
            "Return JSON only. Schema: {schema_version:int, claims:[{claim_id:string, text:string, "
            "citations:[{evidence_id:string, line_start:int, line_end:int, confidence:number}]}], "
            "abstentions:[{claim_id:string, reason:string}]}. "
            "Use evidence IDs exactly as provided (E1, E2...). "
            "Line numbers are 1-indexed and inclusive."
        )
        if errors:
            instructions = (
                f"{instructions} Previous errors: {', '.join(errors)}. "
                "Fix the JSON to satisfy all validation rules."
            )
        return f"{query}\n\n{instructions}"

    def _process_claims_output(self, answer_text: str, evidence: list[EvidenceItem], verifier):
        errors: list[str] = []
        evidence_map = _build_evidence_line_info(evidence)
        evidence_ids = set(evidence_map)
        with otel_span("verification.citations", {"stage_name": "verification.citations"}):
            try:
                parsed = parse_claims_json(answer_text)
            except Exception:
                errors.append("claims_parse_failed")
                return answer_text, [], False, None, errors
            validation = self._claim_validator.validate(parsed.payload, evidence_map=evidence_map)
            if not validation.valid:
                errors.extend(validation.errors)
                return answer_text, [], False, parsed.payload, errors
            claim_objects = [
                Claim(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    evidence_ids=claim.evidence_ids,
                    entity_tokens=claim.entity_tokens,
                )
                for claim in validation.claims
            ]
            verifier_errors = verifier.verify(
                claim_objects,
                valid_evidence=evidence_ids,
                entity_tokens=set(),
            )
            if verifier_errors:
                errors.extend(verifier_errors)
                return answer_text, [], False, parsed.payload, errors
            rendered = render_claims_answer(validation.claims)
            citations = sorted({cid for claim in validation.claims for cid in claim.evidence_ids})
            return rendered, citations, True, parsed.payload, []

    async def _apply_entailment_gate(
        self,
        claims_payload,
        evidence: list[EvidenceItem],
        warnings: list[str],
        *,
        query: str,
        draft_text: str | None,
        retrieve_query: str,
        resolved_time_range,
        filters,
        sanitized: bool,
        aggregates: dict | None,
        final_k: int,
        routing: dict[str, str],
        routing_override: str | None,
        context_pack_format: str,
        context_pack_json: dict,
        context_pack_tron: str | None,
        system_prompt: str,
        budget_manager: BudgetManager,
        budget_state,
        request_id: str | None = None,
    ) -> tuple[str, list[str], object, list[EvidenceItem], dict, str | None] | None:
        with otel_span("verification.entailment", {"stage_name": "verification.entailment"}):
            evidence_by_id = {item.evidence_id: item for item in evidence}
            claims = list(claims_payload.claims)
            max_attempts = int(self._config.verification.entailment.max_attempts)
            attempt = 0
            while True:
                attempt += 1
                heuristic = heuristic_entailment(claims, evidence_by_id)
                verdicts = dict(heuristic.verdicts)
                judge_result = await judge_entailment(
                    self._stage_router,
                    stage=self._config.verification.entailment.judge_stage,
                    claims=claims,
                    evidence_by_id=evidence_by_id,
                )
                for claim_id, verdict in judge_result.verdicts.items():
                    if verdict:
                        verdicts[claim_id] = verdict
                try:
                    claims_payload.entailment = verdicts
                except Exception:
                    pass
                has_contradiction = any(v == "contradicted" for v in verdicts.values())
                has_nei = any(v in {"nei", "not_enough_information"} for v in verdicts.values())
                if has_contradiction:
                    verification_failures_total.labels("entailment", "contradicted").inc()
                    warnings.append("entailment_contradicted")
                    if (
                        self._config.verification.entailment.on_contradiction == "regenerate"
                        and attempt < max_attempts
                    ):
                        regen = await self._regenerate_with_deep_retrieval(
                            query=query,
                            draft_text=draft_text,
                            retrieve_query=retrieve_query,
                            resolved_time_range=resolved_time_range,
                            filters=filters,
                            sanitized=sanitized,
                            aggregates=aggregates,
                            final_k=final_k,
                            routing=routing,
                            routing_override=routing_override,
                            context_pack_format=context_pack_format,
                            system_prompt=system_prompt,
                            budget_manager=budget_manager,
                            budget_state=budget_state,
                            request_id=request_id,
                        )
                        if regen is None:
                            return None
                        (
                            answer_text,
                            citations,
                            claims_payload,
                            evidence,
                            context_pack_json,
                            context_pack_tron,
                        ) = regen
                        evidence_by_id = {item.evidence_id: item for item in evidence}
                        claims = list(claims_payload.claims)
                        continue
                    answer_text = "Answer blocked: contradictions detected in the evidence."
                    return (
                        answer_text,
                        [],
                        claims_payload,
                        evidence,
                        context_pack_json,
                        context_pack_tron,
                    )
                if has_nei:
                    verification_failures_total.labels("entailment", "nei").inc()
                    warnings.append("entailment_nei")
                    if (
                        self._config.verification.entailment.on_nei == "expand_retrieval"
                        and attempt < max_attempts
                    ):
                        regen = await self._regenerate_with_deep_retrieval(
                            query=query,
                            draft_text=draft_text,
                            retrieve_query=retrieve_query,
                            resolved_time_range=resolved_time_range,
                            filters=filters,
                            sanitized=sanitized,
                            aggregates=aggregates,
                            final_k=final_k,
                            routing=routing,
                            routing_override=routing_override,
                            context_pack_format=context_pack_format,
                            system_prompt=system_prompt,
                            budget_manager=budget_manager,
                            budget_state=budget_state,
                            request_id=request_id,
                        )
                        if regen is None:
                            return None
                        (
                            answer_text,
                            citations,
                            claims_payload,
                            evidence,
                            context_pack_json,
                            context_pack_tron,
                        ) = regen
                        evidence_by_id = {item.evidence_id: item for item in evidence}
                        claims = list(claims_payload.claims)
                        continue
                    answer_text = "Not enough evidence to answer."
                    return (
                        answer_text,
                        [],
                        claims_payload,
                        evidence,
                        context_pack_json,
                        context_pack_tron,
                    )
                return None

    async def _regenerate_with_deep_retrieval(
        self,
        *,
        query: str,
        draft_text: str | None,
        retrieve_query: str,
        resolved_time_range,
        filters,
        sanitized: bool,
        aggregates: dict | None,
        final_k: int,
        routing: dict[str, str],
        routing_override: str | None,
        context_pack_format: str,
        system_prompt: str,
        budget_manager: BudgetManager,
        budget_state,
        request_id: str | None = None,
    ) -> tuple[str, list[str], object, list[EvidenceItem], dict, str | None] | None:
        deep_result = self._build_evidence(
            retrieve_query,
            resolved_time_range,
            filters,
            max(final_k, self._config.retrieval.max_k),
            sanitized,
            retrieval_mode="deep",
            request_id=request_id,
        )
        evidence, events, no_evidence = _unpack_evidence_result(deep_result)
        if not evidence or no_evidence:
            return None
        pack = build_context_pack(
            query=query,
            evidence=evidence,
            entity_tokens=self._entities.tokens_for_events(events),
            routing=routing,
            filters={
                "time_range": resolved_time_range,
                "apps": filters.get("app") if filters else None,
                "domains": filters.get("domain") if filters else None,
            },
            sanitized=sanitized,
            aggregates=aggregates,
        )
        context_pack_json = pack.to_json()
        pack_json_text = pack.to_text(extractive_only=False, format="json")
        pack_tron_text = (
            pack.to_text(extractive_only=False, format="tron")
            if context_pack_format == "tron"
            else None
        )
        context_pack_tron = pack_tron_text if context_pack_format == "tron" else None
        provider, decision = self._stage_router.select_llm(
            "final_answer", routing_override=routing_override
        )
        final_query = _build_final_query(query, draft_text)
        if self._requires_claims(decision):
            final_query = self._append_claims_instructions(final_query)
        final_pack = _select_context_pack_text(
            self._config,
            decision,
            context_pack_format,
            pack_json_text,
            pack_tron_text,
            [],
            stage="final_answer",
        )
        answer_text = await provider.generate_answer(
            system_prompt,
            final_query,
            final_pack,
            temperature=decision.temperature,
        )
        verifier = self._get_verifier()
        if self._requires_claims(decision):
            rendered, citations, valid, claims_payload, errors = self._process_claims_output(
                answer_text, evidence, verifier
            )
            if not valid:
                return None
            return (
                rendered,
                citations,
                claims_payload,
                evidence,
                context_pack_json,
                context_pack_tron,
            )
        citations = _extract_citations(answer_text)
        if not _verify_answer(answer_text, citations, evidence, verifier):
            return None
        return answer_text, citations, None, evidence, context_pack_json, context_pack_tron


class _ExtractiveWrapper:
    def compress(self, evidence: list[EvidenceItem]) -> CompressedAnswer:
        return extractive_answer(evidence)


def _stage_max_attempts(decision: object | None) -> int:
    if decision is None:
        return 1
    policy = getattr(decision, "policy", None)
    if policy and getattr(policy, "max_attempts", None):
        try:
            return max(1, int(policy.max_attempts))
        except Exception:
            return 2
    return 2


def _line_offsets(text: str) -> list[int]:
    if text is None:
        return []
    offsets: list[int] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        offsets.append(cursor)
        cursor += len(line)
    if not offsets and text:
        offsets.append(0)
    return offsets


def _build_evidence_line_info(evidence: list[EvidenceItem]) -> dict[str, EvidenceLineInfo]:
    mapping: dict[str, EvidenceLineInfo] = {}
    for item in evidence:
        text = item.text or ""
        lines = text.splitlines()
        if not lines and text:
            lines = [text]
        mapping[item.evidence_id] = EvidenceLineInfo(
            evidence_id=item.evidence_id,
            lines=lines,
            citable=item.citable,
        )
    return mapping


def _extract_citations(answer_text: str) -> list[str]:
    import re

    return re.findall(r"(?:\\[|【)(E\\d+)(?::L\\d+-L\\d+)?(?:\\]|】)", answer_text or "")


def _valid_citations(citations: list[str], evidence: list) -> bool:
    if not citations:
        return False
    evidence_ids = {item.evidence_id for item in evidence}
    return all(citation in evidence_ids for citation in citations)


def _verify_answer(
    answer_text: str,
    citations: list[str],
    evidence: list[EvidenceItem],
    verifier: RulesVerifier,
) -> bool:
    if not _valid_citations(citations, evidence):
        return False
    errors = verifier.verify(
        [
            Claim(
                text=answer_text,
                evidence_ids=citations,
                entity_tokens=[],
            )
        ],
        {item.evidence_id for item in evidence},
        set(),
    )
    return not errors


def _is_confident(score: float, gap: float, config) -> bool:
    return score >= config.fusion_confidence_min and gap >= config.fusion_rank_gap_min


def _build_draft_query(query: str) -> str:
    return (
        f"{query}\n\n"
        "Draft a short answer using the evidence. Include citations like 【E1:L1-L2】."
    )


def _build_final_query(query: str, draft_text: str | None, max_chars: int = 2000) -> str:
    if not draft_text:
        return query
    trimmed = " ".join(draft_text.split())
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars] + "..."
    return (
        f"{query}\n\n"
        "Draft answer to revise:\n"
        f"{trimmed}\n\n"
        "Revise for accuracy and make sure citations are included."
    )


def _select_context_pack_text(
    config: AppConfig,
    decision: object,
    requested_format: str,
    json_text: str,
    tron_text: str | None,
    warnings: list[str],
    *,
    stage: str,
) -> str:
    if requested_format != "tron":
        if not bool(getattr(decision, "cloud", False)):
            return json_text
        if config.output.allow_tron_compression and tron_text:
            warnings.append(f"tron_forced_for_cloud_{stage}")
            return tron_text
        return json_text
    cloud = bool(getattr(decision, "cloud", False))
    if not cloud:
        return tron_text or json_text
    if config.output.allow_tron_compression:
        return tron_text or json_text
    warnings.append(f"tron_disabled_for_cloud_{stage}")
    return json_text


def _build_timeline_answer(evidence: list[EvidenceItem]) -> "CompressedAnswer":
    if not evidence:
        return CompressedAnswer(
            answer="No events found in the requested time range.",
            citations=[],
        )
    sorted_items = sorted(evidence, key=lambda item: item.timestamp)
    lines: list[str] = []
    citations: list[str] = []
    for item in sorted_items:
        snippet = " ".join((item.text or "").split())
        if len(snippet) > 120:
            snippet = snippet[:120] + "..."
        title = item.title or ""
        label = f"{item.app}: {title}".strip(": ")
        if snippet:
            line = f"{item.timestamp} — {label} — {snippet} [{item.evidence_id}]"
        else:
            line = f"{item.timestamp} — {label} [{item.evidence_id}]"
        lines.append(line)
        citations.append(item.evidence_id)
    return CompressedAnswer(answer="\n".join(lines), citations=citations)


def _parse_refined_query(response: str) -> RefinedQueryIntent | None:
    if not response:
        return None
    try:
        data = json.loads(response)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    refined = data.get("refined_query")
    if isinstance(refined, str):
        refined = " ".join(refined.strip().split()) or None
    else:
        refined = None
    time_expression = data.get("time_expression")
    if isinstance(time_expression, str):
        time_expression = " ".join(time_expression.strip().split()) or None
    else:
        time_expression = None
    time_range_payload = data.get("time_range")
    if isinstance(time_range_payload, dict):
        start_iso = time_range_payload.get("start_iso")
        end_iso = time_range_payload.get("end_iso")
        if not (isinstance(start_iso, str) and isinstance(end_iso, str)):
            time_range_payload = None
    else:
        time_range_payload = None
    if not refined and not time_expression and not time_range_payload:
        return None
    return RefinedQueryIntent(
        refined_query=refined,
        time_expression=time_expression,
        time_range_payload=time_range_payload,
    )


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    output: list[str] = []
    for value in values:
        cleaned = (value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def _build_thread_aggregates(
    retrieval: ThreadRetrievalService,
    query: str,
    time_range: tuple[dt.datetime, dt.datetime] | None,
) -> dict:
    if not retrieval:
        return {}
    broad = bool(time_range) or len(query.split()) <= 3
    if not broad:
        return {}
    try:
        candidates = retrieval.retrieve(query, time_range, limit=5)
    except Exception:
        return {}
    if not candidates:
        return {}
    return {
        "threads": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "summary": item.summary,
                "ts_start": item.ts_start.isoformat(),
                "ts_end": item.ts_end.isoformat() if item.ts_end else None,
                "citations": item.citations,
            }
            for item in candidates
        ]
    }


def _merge_aggregates(existing: dict | None, incoming: dict | None) -> dict:
    merged = dict(existing or {})
    for key, value in (incoming or {}).items():
        merged[key] = value
    return merged


def _unpack_evidence_result(
    result: (
        tuple[list[EvidenceItem], list[EventRecord], bool]
        | tuple[list[EvidenceItem], list[EventRecord]]
        | None
    )
) -> tuple[list[EvidenceItem], list[EventRecord], bool]:
    if not result:
        return [], [], True
    if isinstance(result, tuple):
        if len(result) == 3:
            evidence, events, no_evidence = result
            return list(evidence), list(events), bool(no_evidence)
        if len(result) == 2:
            evidence, events = result
            return list(evidence), list(events), not bool(evidence)
    return [], [], True


def _prompt_strategy_payload(provider: object) -> dict | None:
    metadata = getattr(provider, "last_prompt_metadata", None)
    if metadata is None:
        return None
    if hasattr(metadata, "to_dict"):
        return metadata.to_dict()
    return None


def _frame_size_from_tags(tags: dict | None) -> tuple[int, int] | None:
    if not isinstance(tags, dict):
        return None
    meta = tags.get("capture_meta")
    if not isinstance(meta, dict):
        return None
    width = meta.get("frame_width")
    height = meta.get("frame_height")
    try:
        width_val = int(width)
        height_val = int(height)
    except (TypeError, ValueError):
        return None
    if width_val <= 0 or height_val <= 0:
        return None
    return width_val, height_val


def _bbox_norm(bbox: list[int] | None, frame_size: tuple[int, int] | None) -> list[float] | None:
    if not bbox or not frame_size:
        return None
    width, height = frame_size
    if width <= 0 or height <= 0:
        return None
    if len(bbox) >= 8:
        xs = bbox[0::2]
        ys = bbox[1::2]
        if not xs or not ys:
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
    elif len(bbox) >= 4:
        x0, y0, x1, y1 = bbox[:4]
    else:
        return None
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    return [
        round(x0 / width, 6),
        round(y0 / height, 6),
        round(x1 / width, 6),
        round(y1 / height, 6),
    ]


def _no_evidence_message(query: str, has_time_range: bool) -> str:
    _ = query
    if has_time_range:
        return "No evidence found in the selected time range. Try expanding the time range."
    return "No evidence found. Try rephrasing the query or adding a time range."


def _conflict_message(summary: dict | None) -> str:
    if not summary or not summary.get("conflicts"):
        return "Conflicting evidence found. Please review the cited alternatives."
    lines = ["Conflicting evidence found for:"]
    for field, values in summary.get("conflicts", {}).items():
        if not values:
            continue
        formatted = ", ".join(item.get("value", "") for item in values if item.get("value"))
        lines.append(f"- {field}: {formatted}")
    return "\n".join(lines)


def _coverage_thresholds(config: AppConfig) -> tuple[float, float]:
    path = config.next10.tiers_defaults_path
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    coverage = payload.get("coverage", {}) if isinstance(payload.get("coverage"), dict) else {}
    min_normal = float(coverage.get("min_coverage_normal", 0.6))
    min_no_evidence = float(coverage.get("min_coverage_no_evidence", 0.2))
    return min_normal, min_no_evidence


def _build_graph_result(
    answer: str,
    citations: list[str],
    context_pack: dict,
    warnings: list[str],
    *,
    used_llm: bool,
    output_format: str,
    context_pack_tron: str | None,
    prompt_strategy: dict | None = None,
    mode: str = "NORMAL",
    coverage: dict | None = None,
    confidence: dict | None = None,
    budgets: dict | None = None,
    degraded_stages: list[str] | None = None,
    hints: list[dict] | None = None,
    actions: list[dict] | None = None,
    conflict_summary: dict | None = None,
    answer_id: str | None = None,
    query_id: str | None = None,
    claims: dict | None = None,
) -> AnswerGraphResult:
    response_json = None
    response_tron = None
    if output_format in {"json", "tron"}:
        response_json = {
            "answer": answer,
            "citations": citations,
            "warnings": warnings,
            "used_llm": used_llm,
            "mode": mode,
            "coverage": coverage,
            "confidence": confidence,
            "budgets": budgets,
            "degraded_stages": degraded_stages or [],
            "hints": hints or [],
            "actions": actions or [],
            "claims": claims,
            "conflict_summary": conflict_summary,
            "answer_id": answer_id,
            "query_id": query_id,
            "context_pack": context_pack,
            "evidence": build_evidence_payload(context_pack),
        }
        if output_format == "tron":
            from ..format.tron import encode_tron

            response_tron = encode_tron(response_json)
    return AnswerGraphResult(
        answer=answer,
        citations=citations,
        context_pack=context_pack,
        warnings=warnings,
        used_llm=used_llm,
        mode=mode,
        coverage=coverage,
        confidence=confidence,
        budgets=budgets,
        degraded_stages=degraded_stages,
        hints=hints,
        actions=actions,
        conflict_summary=conflict_summary,
        answer_id=answer_id,
        query_id=query_id,
        response_json=response_json,
        response_tron=response_tron,
        context_pack_tron=context_pack_tron,
        prompt_strategy=prompt_strategy,
    )
