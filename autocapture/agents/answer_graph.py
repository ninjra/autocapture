"""Agentic answering pipeline (LangGraph optional)."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from ..config import AppConfig
from ..logging_utils import get_logger
from ..memory.compression import CompressedAnswer, extractive_answer
from ..memory.context_pack import EvidenceItem, build_context_pack, build_evidence_payload
from ..memory.time_intent import (
    is_time_only_expression,
    resolve_time_intent,
    resolve_time_range_for_query,
    resolve_timezone,
)
from ..memory.threads import ThreadRetrievalService
from ..memory.retrieval import RetrieveFilters, RetrievalService, RetrievedEvent
from ..model_ops import StageRouter
from ..storage.models import EventRecord
from ..memory.verification import Claim, RulesVerifier


@dataclass
class AnswerGraphResult:
    answer: str
    citations: list[str]
    context_pack: dict
    warnings: list[str]
    used_llm: bool
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
    ) -> None:
        self._config = config
        self._retrieval = retrieval
        self._prompt_registry = prompt_registry
        self._entities = entities
        self._log = get_logger("answer_graph")
        self._stage_router = StageRouter(config)
        self._thread_retrieval = ThreadRetrievalService(
            config,
            retrieval._db,  # type: ignore[attr-defined]
            embedder=getattr(retrieval, "_embedder", None),
            vector_index=getattr(retrieval, "_vector", None),
        )

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
        speculative = bool(
            self._config.retrieval.speculative_enabled
            and self._config.model_stages.draft_generate.enabled
        )
        draft_k = self._config.retrieval.speculative_draft_k if speculative else k
        final_k = self._config.retrieval.speculative_final_k if speculative else k
        retrieval_mode = "baseline" if speculative else None
        evidence, events = self._build_evidence(
            retrieve_query,
            resolved_time_range,
            filters,
            draft_k,
            sanitized,
            retrieval_mode=retrieval_mode,
        )
        thread_aggregates = _build_thread_aggregates(
            self._thread_retrieval,
            normalized_query,
            resolved_time_range,
        )
        aggregates = _merge_aggregates(aggregates, thread_aggregates)
        if not evidence:
            warnings.append("no_evidence")
        if (
            not time_only
            and self._should_refine(evidence)
            and self._config.model_stages.query_refine.enabled
            and not speculative
        ):
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
                evidence, events = self._build_evidence(
                    refined_query,
                    resolved_time_range,
                    filters,
                    draft_k,
                    sanitized,
                    retrieval_mode=retrieval_mode,
                )
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
            return _build_graph_result(
                timeline.answer,
                timeline.citations,
                context_pack_json,
                warnings,
                used_llm=False,
                output_format=output_format,
                context_pack_tron=context_pack_tron,
                prompt_strategy=None,
            )
        if extractive_only:
            compressed = extractive_answer(evidence)
            return _build_graph_result(
                compressed.answer,
                compressed.citations,
                context_pack_json,
                warnings,
                used_llm=False,
                output_format=output_format,
                context_pack_tron=context_pack_tron,
                prompt_strategy=None,
            )

        system_prompt = self._prompt_registry.get("ANSWER_WITH_CONTEXT_PACK").system_prompt
        draft_text: str | None = None
        draft_citations: list[str] = []
        draft_valid = False
        speculative_confident = False
        if speculative:
            score, gap = self._evidence_confidence(evidence)
            speculative_confident = _is_confident(score, gap, self._config.retrieval)
        if self._config.model_stages.draft_generate.enabled:
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
                draft_text = await draft_provider.generate_answer(
                    system_prompt,
                    draft_query,
                    draft_pack,
                    temperature=draft_decision.temperature,
                )
                draft_citations = _extract_citations(draft_text)
                draft_valid = _verify_answer(draft_text, draft_citations, evidence)
            except Exception as exc:
                warnings.append("draft_failed")
                self._log.warning("Draft generation failed; skipping: {}", exc)

        if speculative and draft_valid and draft_text is not None and speculative_confident:
            return _build_graph_result(
                draft_text,
                draft_citations,
                context_pack_json,
                warnings,
                used_llm=True,
                output_format=output_format,
                context_pack_tron=context_pack_tron,
                prompt_strategy=None,
            )

        if speculative:
            evidence, events = self._build_evidence(
                retrieve_query,
                resolved_time_range,
                filters,
                final_k,
                sanitized,
                retrieval_mode="deep",
            )
            if not evidence:
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
        try:
            final_provider, final_decision = self._stage_router.select_llm(
                "final_answer", routing_override=routing_override
            )
            final_query = _build_final_query(query, draft_text)
            final_pack = _select_context_pack_text(
                self._config,
                final_decision,
                context_pack_format,
                pack_json_text,
                pack_tron_text,
                warnings,
                stage="final_answer",
            )
            answer_text = await final_provider.generate_answer(
                system_prompt,
                final_query,
                final_pack,
                temperature=final_decision.temperature,
            )
            prompt_strategy_payload = _prompt_strategy_payload(final_provider)
            citations = _extract_citations(answer_text)
            final_valid = _verify_answer(answer_text, citations, evidence)
        except Exception as exc:
            warnings.append("final_failed")
            self._log.warning("Final answer failed; falling back: {}", exc)

        if final_valid and (not draft_valid or len(citations) >= len(draft_citations)):
            used_llm = True
        elif draft_valid and draft_text is not None:
            answer_text = draft_text
            citations = draft_citations
            used_llm = True
        else:
            compressed = extractive_answer(evidence)
            answer_text = compressed.answer
            citations = compressed.citations
            used_llm = False

        return _build_graph_result(
            answer_text,
            citations,
            context_pack_json,
            warnings,
            used_llm=used_llm,
            output_format=output_format,
            context_pack_tron=context_pack_tron,
            prompt_strategy=prompt_strategy_payload,
        )

    def _build_evidence(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: dict[str, list[str]] | None,
        k: int,
        sanitized: bool,
        retrieval_mode: str | None = None,
    ):
        retrieve_filters = None
        if filters:
            retrieve_filters = RetrieveFilters(
                apps=filters.get("app"), domains=filters.get("domain")
            )
        results = self._retrieval.retrieve(
            query, time_range, retrieve_filters, limit=k, mode=retrieval_mode
        )
        if not results:
            with self._retrieval._db.session() as session:  # type: ignore[attr-defined]
                stmt = select(EventRecord).where(EventRecord.ocr_text.ilike(f"%{query}%"))
                if time_range:
                    stmt = stmt.where(EventRecord.ts_start.between(*time_range))
                events = session.execute(stmt).scalars().all()
            results = [RetrievedEvent(event=event, score=0.5) for event in events[:k]]
        if not results and time_range and not query:
            results = self._retrieval.retrieve("", time_range, retrieve_filters, limit=k)
        if not results:
            fallback_events = list(self._retrieval.list_events(limit=k))
            results = [RetrievedEvent(event=event, score=0.3) for event in fallback_events]
        evidence: list[EvidenceItem] = []
        events = []
        for idx, result in enumerate(results, start=1):
            event = result.event
            events.append(event)
            snippet = (event.ocr_text or "")[:500]
            app_name = event.app_name
            title = event.window_title
            domain = event.domain
            if sanitized:
                snippet = self._entities.pseudonymize_text(snippet)
                app_name = self._entities.pseudonymize_text(app_name)
                title = self._entities.pseudonymize_text(title)
                if domain:
                    domain = self._entities.pseudonymize_text(domain)
            evidence.append(
                EvidenceItem(
                    evidence_id=f"E{idx}",
                    event_id=event.event_id,
                    timestamp=event.ts_start.isoformat(),
                    ts_end=event.ts_end.isoformat() if event.ts_end else None,
                    app=app_name,
                    title=title,
                    domain=domain,
                    score=result.score,
                    spans=[],
                    text=snippet,
                    screenshot_path=event.screenshot_path,
                    screenshot_hash=event.screenshot_hash,
                    retrieval={
                        "engine": getattr(result, "engine", "hybrid"),
                        "rank": getattr(result, "rank", idx),
                        "rank_gap": getattr(result, "rank_gap", 0.0),
                        "lexical_score": getattr(result, "lexical_score", 0.0),
                        "vector_score": getattr(result, "vector_score", 0.0),
                        "sparse_score": getattr(result, "sparse_score", 0.0),
                        "late_score": getattr(result, "late_score", 0.0),
                        "matched_spans": getattr(result, "matched_span_keys", []),
                        "ts_start": event.ts_start.isoformat(),
                    },
                )
            )
        return evidence, events

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
            response = await provider.generate_answer(
                prompt.system_prompt,
                query,
                context,
                temperature=decision.temperature,
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


def _extract_citations(answer_text: str) -> list[str]:
    import re

    return re.findall(r"E\d+", answer_text or "")


def _valid_citations(citations: list[str], evidence: list) -> bool:
    if not citations:
        return False
    evidence_ids = {item.evidence_id for item in evidence}
    return all(citation in evidence_ids for citation in citations)


def _verify_answer(answer_text: str, citations: list[str], evidence: list[EvidenceItem]) -> bool:
    if not _valid_citations(citations, evidence):
        return False
    verifier = RulesVerifier()
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
    return f"{query}\n\n" "Draft a short answer using the evidence. Include citations like [E1]."


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


def _prompt_strategy_payload(provider: object) -> dict | None:
    metadata = getattr(provider, "last_prompt_metadata", None)
    if metadata is None:
        return None
    if hasattr(metadata, "to_dict"):
        return metadata.to_dict()
    return None


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
) -> AnswerGraphResult:
    response_json = None
    response_tron = None
    if output_format in {"json", "tron"}:
        response_json = {
            "answer": answer,
            "citations": citations,
            "warnings": warnings,
            "used_llm": used_llm,
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
        response_json=response_json,
        response_tron=response_tron,
        context_pack_tron=context_pack_tron,
        prompt_strategy=prompt_strategy,
    )
