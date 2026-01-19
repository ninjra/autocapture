"""Provenance chain verification gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from autocapture.next10.harness import build_harness
from autocapture.storage.ledger import LedgerWriter


def _write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    corpus, _retrieval, answer_graph = build_harness(enable_rerank=True)
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
    answer_id = result.answer_id or ""
    ledger = LedgerWriter(corpus.db)
    chain_ok = ledger.validate_chain(answer_id)
    report = {"answer_id": answer_id, "chain_valid": chain_ok}
    output = Path("artifacts") / "provenance_report.json"
    _write_report(report, output)
    if not chain_ok:
        print("Provenance gate failed; see artifacts/provenance_report.json")
        return 2
    print("Provenance gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
