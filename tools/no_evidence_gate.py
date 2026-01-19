"""No-evidence determinism gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from autocapture.next10.harness import build_harness


def _write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    corpus, _retrieval, answer_graph = build_harness(enable_rerank=False)
    result = asyncio.run(
        answer_graph.run(
            "no evidence query",
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
    report = {"mode": result.mode, "answer": result.answer}
    output = Path("artifacts") / "no_evidence_report.json"
    _write_report(report, output)
    if result.mode != "NO_EVIDENCE":
        print("No-evidence gate failed; see artifacts/no_evidence_report.json")
        return 2
    print("No-evidence gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
