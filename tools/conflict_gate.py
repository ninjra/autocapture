"""Conflict scenario gate."""

from __future__ import annotations

import json
from pathlib import Path

from autocapture.answer.conflict import detect_conflicts
from autocapture.memory.context_pack import EvidenceItem, EvidenceSpan


def _write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    timestamp = "2026-01-17T12:00:00+00:00"
    span = EvidenceSpan(span_id="S1", start=0, end=10, conf=0.9, bbox=[0, 0, 1, 1])
    evidence = [
        EvidenceItem(
            evidence_id="E1",
            event_id="EVT-1",
            timestamp=timestamp,
            ts_end=None,
            app="App",
            title="Title",
            domain=None,
            score=0.9,
            spans=[span],
            text="Project Alpha cost $100",
            screenshot_path=None,
            screenshot_hash=None,
            retrieval=None,
        ),
        EvidenceItem(
            evidence_id="E2",
            event_id="EVT-2",
            timestamp=timestamp,
            ts_end=None,
            app="App",
            title="Title",
            domain=None,
            score=0.85,
            spans=[span],
            text="Project Alpha cost $150",
            screenshot_path=None,
            screenshot_hash=None,
            retrieval=None,
        ),
    ]
    result = detect_conflicts(evidence)
    report = {"conflict": result.conflict, "changed_over_time": result.changed_over_time}
    output = Path("artifacts") / "conflict_report.json"
    _write_report(report, output)
    if not result.conflict:
        print("Conflict gate failed; see artifacts/conflict_report.json")
        return 2
    print("Conflict gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
