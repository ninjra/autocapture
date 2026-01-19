"""Latency budget regression gate."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from autocapture.next10.harness import build_harness
from autocapture.runtime_budgets import BudgetManager


def _write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    corpus, _retrieval, answer_graph = build_harness(enable_rerank=True)
    budgets = BudgetManager(corpus.config)
    start = time.monotonic()
    result = asyncio.run(
        answer_graph.run(
            "alpha cost",
            time_range=None,
            filters=None,
            k=4,
            sanitized=True,
            extractive_only=True,
            routing={"llm": "disabled"},
            output_format="text",
            context_pack_format="json",
        )
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    snapshot = budgets.snapshot()
    report = {
        "total_ms": elapsed_ms,
        "budget_total_ms": snapshot.total_ms,
        "mode": result.mode,
    }
    output = Path("artifacts") / "latency_report.json"
    _write_report(report, output)
    if elapsed_ms > snapshot.total_ms * 2:
        print("Latency gate failed; see artifacts/latency_report.json")
        return 2
    print("Latency gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
