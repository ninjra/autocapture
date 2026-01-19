"""Citation integrity simulation gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from autocapture.next10.harness import build_harness
from autocapture.storage.models import FrameRecord


def _write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    corpus, _retrieval, answer_graph = build_harness(enable_rerank=True)
    first = asyncio.run(
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
    with corpus.db.session() as session:
        frames = session.query(FrameRecord).all()
    for frame in frames:
        if frame.media_path:
            try:
                Path(frame.media_path).unlink(missing_ok=True)
            except Exception:
                pass

    second = asyncio.run(
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
    report = {
        "first_mode": first.mode,
        "second_mode": second.mode,
        "warnings": second.warnings,
    }
    output = Path("artifacts") / "integrity_report.json"
    _write_report(report, output)
    if second.mode != "NO_EVIDENCE":
        print("Integrity gate failed; see artifacts/integrity_report.json")
        return 2
    print("Integrity gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
