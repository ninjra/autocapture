"""Retrieval sensitivity +/-1 gate."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from sqlalchemy import select

from autocapture.next10.harness import build_harness
from autocapture.storage.models import RetrievalHitRecord
from autocapture.memory.retrieval import RetrievalService


def _write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _load_tiers(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_tiers(payload: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix="-tiers.json")
    json.dump(payload, tmp)
    tmp.flush()
    return Path(tmp.name)


def _top_span_ids(db, query_id: str) -> list[str]:
    with db.session() as session:
        hits = (
            session.execute(
                select(RetrievalHitRecord).where(RetrievalHitRecord.query_id == query_id)
            )
            .scalars()
            .all()
        )
    if not hits:
        return []
    tier_order = {"RERANK": 3, "FUSION": 2, "FAST": 1}
    best_tier = max((tier_order.get(hit.tier, 0) for hit in hits), default=0)
    selected = [hit for hit in hits if tier_order.get(hit.tier, 0) == best_tier]
    selected.sort(key=lambda hit: hit.rank)
    return [hit.span_id for hit in selected if hit.span_id]


async def _run_answer(answer_graph, query: str) -> dict:
    result = await answer_graph.run(
        query,
        time_range=None,
        filters=None,
        k=4,
        sanitized=True,
        extractive_only=True,
        routing={"llm": "disabled"},
        output_format="text",
        context_pack_format="json",
    )
    return {"mode": result.mode, "coverage": result.coverage or {}}


def _jaccard(a: list[str], b: list[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def main() -> int:
    corpus, retrieval, answer_graph = build_harness(enable_rerank=True)
    base_tiers = _load_tiers(corpus.config.next10.tiers_defaults_path)

    queries = list(corpus.query_map.keys()) + ["no evidence query"]
    base_results: dict[str, dict] = {}
    for query in queries:
        batch = retrieval.retrieve_tiered(query, None, None, limit=4)
        base_results[query] = {
            "span_ids": _top_span_ids(corpus.db, batch.query_id or ""),
            "answer": asyncio.run(_run_answer(answer_graph, query)),
        }

    variants = []
    for key, delta in [("k_lex", -1), ("k_lex", 1), ("k_vec", -1), ("k_vec", 1), ("top_n", -1), ("top_n", 1)]:
        variant = json.loads(json.dumps(base_tiers))
        if key in {"k_lex", "k_vec"}:
            fast = variant.setdefault("fast", {})
            fast[key] = max(1, int(fast.get(key, 8)) + delta)
        else:
            rerank = variant.setdefault("rerank", {})
            rerank[key] = max(1, int(rerank.get(key, 8)) + delta)
        variants.append(variant)

    jaccards: list[float] = []
    mode_flips = 0
    coverage_deltas: list[float] = []

    for variant in variants:
        variant_path = _write_tiers(variant)
        corpus.config.next10.tiers_defaults_path = variant_path
        variant_retrieval = RetrievalService(corpus.db, corpus.config)
        variant_answer_graph = type(answer_graph)(
            corpus.config,
            variant_retrieval,
            prompt_registry=answer_graph._prompt_registry,
            entities=answer_graph._entities,
        )
        for query in queries:
            batch = variant_retrieval.retrieve_tiered(query, None, None, limit=4)
            span_ids = _top_span_ids(corpus.db, batch.query_id or "")
            base = base_results[query]
            jaccards.append(_jaccard(base["span_ids"], span_ids))
            variant_answer = asyncio.run(_run_answer(variant_answer_graph, query))
            if variant_answer["mode"] != base["answer"]["mode"]:
                mode_flips += 1
            base_cov = float(base["answer"].get("coverage", {}).get("sentence_coverage", 0.0))
            variant_cov = float(variant_answer.get("coverage", {}).get("sentence_coverage", 0.0))
            coverage_deltas.append(abs(variant_cov - base_cov))

    mean_jaccard = sum(jaccards) / max(len(jaccards), 1)
    mode_flip_rate = mode_flips / max(len(queries) * len(variants), 1)
    coverage_delta_mean = sum(coverage_deltas) / max(len(coverage_deltas), 1)

    report = {
        "mean_jaccard": mean_jaccard,
        "mode_flip_rate": mode_flip_rate,
        "coverage_delta_mean": coverage_delta_mean,
        "variants": len(variants),
    }
    output = Path("artifacts") / "instability_report.json"
    _write_report(report, output)

    if mean_jaccard < 0.4 or mode_flip_rate > 0.5:
        print("Retrieval sensitivity gate failed; see artifacts/instability_report.json")
        return 2
    print("Retrieval sensitivity gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
