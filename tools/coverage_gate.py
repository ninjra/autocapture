"""Coverage regression gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from autocapture.next10.harness import build_harness


def _load_thresholds(config_path: Path) -> float:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return 0.5
    coverage = payload.get("coverage", {}) if isinstance(payload, dict) else {}
    return float(coverage.get("min_coverage_normal", 0.5))


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
    coverage = result.coverage or {}
    threshold = _load_thresholds(corpus.config.next10.tiers_defaults_path)
    report = {
        "sentence_coverage": coverage.get("sentence_coverage", 0.0),
        "evidence_count": coverage.get("evidence_count", 0),
        "threshold": threshold,
        "mode": result.mode,
    }
    output = Path("artifacts") / "coverage_report.json"
    _write_report(report, output)
    if report["sentence_coverage"] < threshold and report["mode"] != "NO_EVIDENCE":
        print("Coverage gate failed; see artifacts/coverage_report.json")
        return 2
    print("Coverage gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
