"""Agentic answering pipeline (LangGraph optional)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from ..config import AppConfig
from ..memory.compression import extractive_answer
from ..memory.context_pack import EvidenceItem, build_context_pack
from ..memory.retrieval import RetrieveFilters, RetrievalService, RetrievedEvent
from ..memory.router import ProviderRouter
from ..storage.models import EventRecord
from ..memory.verification import Claim, RulesVerifier


@dataclass
class AnswerGraphResult:
    answer: str
    citations: list[str]
    context_pack: dict
    warnings: list[str]
    used_llm: bool


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
        aggregates: dict | None = None,
    ) -> AnswerGraphResult:
        warnings: list[str] = []
        normalized_query = self._normalize_query(query)
        evidence, events = self._build_evidence(normalized_query, time_range, filters, k, sanitized)
        if not evidence:
            warnings.append("no_evidence")
        if self._should_refine(evidence):
            refined = self._refine_query(normalized_query, evidence)
            if refined and refined != normalized_query:
                evidence, events = self._build_evidence(refined, time_range, filters, k, sanitized)
        pack = build_context_pack(
            query=query,
            evidence=evidence,
            entity_tokens=self._entities.tokens_for_events(events),
            routing=routing,
            filters={
                "time_range": time_range,
                "apps": filters.get("app") if filters else None,
                "domains": filters.get("domain") if filters else None,
            },
            sanitized=sanitized,
            aggregates=aggregates,
        )
        if extractive_only:
            compressed = extractive_answer(evidence)
            return AnswerGraphResult(
                answer=compressed.answer,
                citations=compressed.citations,
                context_pack=pack.to_json(),
                warnings=warnings,
                used_llm=False,
            )
        provider, _decision = ProviderRouter(
            self._merge_routing(routing),
            self._config.llm,
            offline=self._config.offline,
            privacy=self._config.privacy,
        ).select_llm()
        system_prompt = self._prompt_registry.get("ANSWER_WITH_CONTEXT_PACK").system_prompt
        answer_text = await provider.generate_answer(
            system_prompt,
            query,
            pack.to_text(extractive_only=False),
        )
        citations = _extract_citations(answer_text)
        used_llm = False
        if not _valid_citations(citations, evidence):
            compressed = extractive_answer(evidence)
            answer_text = compressed.answer
            citations = compressed.citations
            used_llm = False
        else:
            verifier = RulesVerifier()
            verifier.verify(
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
            used_llm = True
        return AnswerGraphResult(
            answer=answer_text,
            citations=citations,
            context_pack=pack.to_json(),
            warnings=warnings,
            used_llm=used_llm,
        )

    def _build_evidence(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: dict[str, list[str]] | None,
        k: int,
        sanitized: bool,
    ):
        retrieve_filters = None
        if filters:
            retrieve_filters = RetrieveFilters(
                apps=filters.get("app"), domains=filters.get("domain")
            )
        results = self._retrieval.retrieve(query, time_range, retrieve_filters, limit=k)
        if not results:
            with self._retrieval._db.session() as session:  # type: ignore[attr-defined]
                stmt = select(EventRecord).where(EventRecord.ocr_text.ilike(f"%{query}%"))
                if time_range:
                    stmt = stmt.where(EventRecord.ts_start.between(*time_range))
                events = session.execute(stmt).scalars().all()
            results = [RetrievedEvent(event=event, score=0.5) for event in events[:k]]
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
                    app=app_name,
                    title=title,
                    domain=domain,
                    score=result.score,
                    spans=[],
                    text=snippet,
                )
            )
        return evidence, events

    def _normalize_query(self, query: str) -> str:
        return query.strip()

    def _should_refine(self, evidence: list) -> bool:
        return len(evidence) < 3

    def _refine_query(self, query: str, evidence: list) -> str:
        tokens = []
        for item in evidence:
            app = getattr(item, "app", None)
            if app:
                tokens.append(app)
        if not tokens:
            return query
        return f"{query} {' '.join(tokens[:3])}"

    def _merge_routing(self, routing: dict[str, str]) -> Any:
        from ..config import ProviderRoutingConfig

        return ProviderRoutingConfig(**routing)


def _extract_citations(answer_text: str) -> list[str]:
    import re

    return re.findall(r"E\\d+", answer_text or "")


def _valid_citations(citations: list[str], evidence: list) -> bool:
    if not citations:
        return False
    evidence_ids = {item.evidence_id for item in evidence}
    return all(citation in evidence_ids for citation in citations)
