from __future__ import annotations

import json
from pathlib import Path

from autocapture.bench.timing import TimingTracer


def test_timing_tracer_redacts_fields(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    with TimingTracer(enabled=True, file_path=trace_path, redact=True) as tracer:
        with tracer.span("phase", case_id="case1", secret="super-secret", count=3):
            pass

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert lines
    payload = json.loads(lines[0])
    assert payload["phase"] == "phase"
    assert payload["fields"]["case_id"] == "case1"
    assert payload["fields"]["secret"] == "[REDACTED]"
    assert payload["fields"]["count"] == 3
