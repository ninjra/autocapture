"""PromptOps evaluation helpers with detailed case results."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config import AppConfig
from ..evals import EvalMetrics, _build_evidence, _extract_citations, _load_prompt_registry
from ..llm.providers import LLMProvider
from ..llm.prompt_strategy import PromptStrategySettings
from ..logging_utils import get_logger
from ..memory.compression import extractive_answer
from ..memory.context_pack import build_context_pack
from ..memory.entities import EntityResolver, SecretStore
from ..memory.retrieval import RetrievalService
from ..memory.router import ProviderRouter
from ..memory.verification import Claim, RulesVerifier
from ..storage.database import DatabaseManager
from ..security.token_vault import TokenVaultStore


@dataclass(frozen=True)
class EvalCaseResult:
    query: str
    refused: bool
    citations_found: list[str]
    citation_hit: bool
    verifier_pass: bool
    verifier_errors: list[str]
    latency_ms: float


@dataclass(frozen=True)
class EvalRunResult:
    metrics: EvalMetrics
    cases: list[EvalCaseResult]


def run_eval_detailed(
    config: AppConfig,
    eval_path: Path,
    *,
    overrides: Iterable[object] | None = None,
    llm_provider: LLMProvider | None = None,
) -> EvalRunResult:
    log = get_logger("evals")
    db = DatabaseManager(config.database)
    retrieval = RetrievalService(db, config)
    secret = SecretStore(Path(config.capture.data_dir)).get_or_create()
    entities = EntityResolver(db, secret, token_vault=TokenVaultStore(config, db))
    verifier = RulesVerifier()
    prompts = _load_prompt_registry(config, overrides)

    items = json.loads(eval_path.read_text(encoding="utf-8"))
    total = len(items)
    citation_hits = 0
    verifier_pass = 0
    refusals = 0
    latencies: list[float] = []
    cases: list[EvalCaseResult] = []

    provider = llm_provider
    if provider is None:
        provider = ProviderRouter(
            config.routing,
            config.llm,
            config=config,
            offline=config.offline,
            privacy=config.privacy,
            prompt_strategy=PromptStrategySettings.from_llm_config(
                config.llm, data_dir=config.capture.data_dir
            ),
        ).select_llm()[0]

    for item in items:
        query = item["query"]
        evidence, events, _no_evidence = _build_evidence(
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
                    priority="background",
                )
            )
        except Exception as exc:
            log.warning("Eval LLM failed; using extractive answer: {}", exc)
            compressed = extractive_answer(evidence)
            answer_text = compressed.answer
        latency = (time.monotonic() - start) * 1000
        latencies.append(latency)

        citations = _extract_citations(answer_text)
        citation_hit = bool(citations)
        if citation_hit:
            citation_hits += 1
        refused = "not enough evidence" in answer_text.lower()
        if refused:
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
        verifier_ok = not errors
        if verifier_ok:
            verifier_pass += 1
        cases.append(
            EvalCaseResult(
                query=query,
                refused=refused,
                citations_found=citations,
                citation_hit=citation_hit,
                verifier_pass=verifier_ok,
                verifier_errors=errors,
                latency_ms=latency,
            )
        )

    if total == 0:
        metrics = EvalMetrics(
            citation_coverage=0.0,
            verifier_pass_rate=0.0,
            refusal_rate=0.0,
            mean_latency_ms=0.0,
        )
    else:
        metrics = EvalMetrics(
            citation_coverage=citation_hits / total,
            verifier_pass_rate=verifier_pass / total,
            refusal_rate=refusals / total,
            mean_latency_ms=sum(latencies) / max(len(latencies), 1),
        )
    return EvalRunResult(metrics=metrics, cases=cases)


def run_eval(
    config: AppConfig,
    eval_path: Path,
    *,
    overrides: Iterable[object] | None = None,
    llm_provider: LLMProvider | None = None,
) -> EvalMetrics:
    return run_eval_detailed(
        config, eval_path, overrides=overrides, llm_provider=llm_provider
    ).metrics
