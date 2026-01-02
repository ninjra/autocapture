"""Evaluation harness for context pack and answer quality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .memory.compression import extractive_answer
from .memory.context_pack import build_context_pack
from .memory.entities import EntityResolver, SecretStore
from .memory.retrieval import RetrievalService
from .memory.verification import Claim, RulesVerifier
from .storage.database import DatabaseManager


@dataclass(frozen=True)
class EvalMetrics:
    citation_coverage: float
    verifier_pass_rate: float


def run_eval(config: AppConfig, eval_path: Path) -> EvalMetrics:
    db = DatabaseManager(config.database)
    retrieval = RetrievalService(db)
    secret = SecretStore(Path(config.capture.data_dir)).get_or_create()
    entities = EntityResolver(db, secret)
    verifier = RulesVerifier()

    items = json.loads(eval_path.read_text(encoding="utf-8"))
    total = len(items)
    citation_hits = 0
    verifier_pass = 0

    for item in items:
        evidence = retrieval.retrieve(item["query"], None, None, limit=5)
        evidence_items = [result.event for result in evidence]
        pack = build_context_pack(
            query=item["query"],
            evidence=[],
            entity_tokens=entities.tokens_for_events(evidence_items),
            routing={"llm": "local"},
            filters={"time_range": None, "apps": None, "domains": None},
            sanitized=True,
        )
        compressed = extractive_answer([])
        if "[" in compressed.answer or compressed.citations:
            citation_hits += 1
        claims = [
            Claim(text=compressed.answer, evidence_ids=compressed.citations, entity_tokens=[])
        ]
        errors = verifier.verify(
            claims, valid_evidence=set(compressed.citations), entity_tokens=set()
        )
        if not errors:
            verifier_pass += 1

    if total == 0:
        return EvalMetrics(citation_coverage=0.0, verifier_pass_rate=0.0)
    return EvalMetrics(
        citation_coverage=citation_hits / total,
        verifier_pass_rate=verifier_pass / total,
    )
