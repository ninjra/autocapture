"""Evaluation harness for context pack and answer quality."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .config import AppConfig
from .llm.providers import LLMProvider
from .logging_utils import get_logger
from .memory.compression import extractive_answer
from .memory.context_pack import EvidenceItem, EvidenceSpan, build_context_pack
from .memory.entities import EntityResolver, SecretStore
from .memory.prompts import PromptRegistry, PromptTemplate
from .memory.retrieval import RetrievalService
from .memory.router import ProviderRouter
from .memory.verification import Claim, RulesVerifier
from .storage.database import DatabaseManager


@dataclass(frozen=True)
class EvalMetrics:
    citation_coverage: float
    verifier_pass_rate: float
    refusal_rate: float
    mean_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "citation_coverage": self.citation_coverage,
            "verifier_pass_rate": self.verifier_pass_rate,
            "refusal_rate": self.refusal_rate,
            "mean_latency_ms": self.mean_latency_ms,
        }


def run_eval(
    config: AppConfig,
    eval_path: Path,
    *,
    overrides: Iterable[object] | None = None,
    llm_provider: LLMProvider | None = None,
) -> EvalMetrics:
    log = get_logger("evals")
    db = DatabaseManager(config.database)
    retrieval = RetrievalService(db, config)
    secret = SecretStore(Path(config.capture.data_dir)).get_or_create()
    entities = EntityResolver(db, secret)
    verifier = RulesVerifier()
    prompts = _load_prompt_registry(overrides)

    items = json.loads(eval_path.read_text(encoding="utf-8"))
    total = len(items)
    citation_hits = 0
    verifier_pass = 0
    refusals = 0
    latencies: list[float] = []

    provider = llm_provider
    if provider is None:
        provider = ProviderRouter(
            config.routing, config.llm, offline=config.offline, privacy=config.privacy
        ).select_llm()[0]

    for item in items:
        query = item["query"]
        evidence, events = _build_evidence(
            retrieval, entities, query, limit=5, sanitized=True
        )
        pack = build_context_pack(
            query=query,
            evidence=evidence,
            entity_tokens=entities.tokens_for_events(events),
            routing={"llm": config.routing.llm},
            filters={"time_range": None, "apps": None, "domains": None},
            sanitized=True,
        )
        start = time.monotonic()
        try:
            system_prompt = prompts.get("ANSWER_WITH_CONTEXT_PACK").system_prompt
            answer_text = asyncio.run(
                provider.generate_answer(
                    system_prompt,
                    query,
                    pack.to_text(extractive_only=False),
                )
            )
        except Exception as exc:
            log.warning("Eval LLM failed; using extractive answer: {}", exc)
            compressed = extractive_answer(evidence)
            answer_text = compressed.answer
        latency = (time.monotonic() - start) * 1000
        latencies.append(latency)

        citations = _extract_citations(answer_text)
        if citations:
            citation_hits += 1
        if "not enough evidence" in answer_text.lower():
            refusals += 1
        claims = [
            Claim(
                text=answer_text,
                evidence_ids=citations,
                entity_tokens=[],
            )
        ]
        errors = verifier.verify(
            claims,
            valid_evidence={item.evidence_id for item in evidence},
            entity_tokens=set(),
        )
        if not errors:
            verifier_pass += 1

    if total == 0:
        return EvalMetrics(
            citation_coverage=0.0,
            verifier_pass_rate=0.0,
            refusal_rate=0.0,
            mean_latency_ms=0.0,
        )
    return EvalMetrics(
        citation_coverage=citation_hits / total,
        verifier_pass_rate=verifier_pass / total,
        refusal_rate=refusals / total,
        mean_latency_ms=sum(latencies) / max(len(latencies), 1),
    )


def _load_prompt_registry(overrides: Iterable[object] | None) -> PromptRegistry:
    registry = PromptRegistry.from_package("autocapture.prompts.derived")
    registry.load()
    if not overrides:
        return registry
    for proposal in overrides:
        if not hasattr(proposal, "derived_content"):
            continue
        data = yaml.safe_load(proposal.derived_content)
        name = data["name"]
        registry._cache[name] = PromptTemplate(
            name=name,
            version=data["version"],
            system_prompt=data["system_prompt"],
            tags=data.get("tags", []),
            raw_template=data.get("raw_template", data["system_prompt"]),
            derived_template=data.get("derived_template", data["system_prompt"]),
        )
    return registry


def _build_evidence(
    retrieval: RetrievalService,
    entities: EntityResolver,
    query: str,
    *,
    limit: int,
    sanitized: bool,
) -> tuple[list[EvidenceItem], list[object]]:
    results = retrieval.retrieve(query, None, None, limit=limit)
    evidence: list[EvidenceItem] = []
    events: list[object] = []
    for idx, result in enumerate(results, start=1):
        event = result.event
        events.append(event)
        snippet, snippet_offset = _snippet_for_query(event.ocr_text or "", query)
        spans = []
        if snippet:
            spans.append(
                EvidenceSpan(
                    span_id=result.event.event_id,
                    start=snippet_offset,
                    end=snippet_offset + len(snippet),
                    conf=0.5,
                )
            )
        app_name = event.app_name
        title = event.window_title
        domain = event.domain
        if sanitized:
            snippet = entities.pseudonymize_text(snippet)
            app_name = entities.pseudonymize_text(app_name)
            title = entities.pseudonymize_text(title)
            if domain:
                domain = entities.pseudonymize_text(domain)
        evidence.append(
            EvidenceItem(
                evidence_id=f"E{idx}",
                event_id=event.event_id,
                timestamp=event.ts_start.isoformat(),
                app=app_name,
                title=title,
                domain=domain,
                score=result.score,
                spans=spans,
                text=snippet,
            )
        )
    return evidence, events


def _snippet_for_query(text: str, query: str, window: int = 200) -> tuple[str, int]:
    if not text:
        return "", 0
    lower = text.lower()
    q = query.lower()
    idx = lower.find(q)
    if idx == -1:
        return text[:window], 0
    start = max(0, idx - window // 2)
    end = min(len(text), idx + window // 2)
    return text[start:end], start


_CITATION_RE = re.compile(r"\[(E\d+)\]")


def _extract_citations(answer: str) -> list[str]:
    if not answer:
        return []
    return list(dict.fromkeys(_CITATION_RE.findall(answer)))
