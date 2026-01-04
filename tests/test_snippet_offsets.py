from __future__ import annotations

from autocapture.api.server import _snippet_for_query, _spans_for_event


def test_snippet_offsets_keep_spans_in_bounds() -> None:
    text = "Alpha beta gamma delta epsilon zeta eta theta iota"
    spans = [
        {"span_id": "S1", "span_key": "S1", "text": "gamma", "start": 12, "end": 17, "conf": 0.9},
        {"span_id": "S2", "span_key": "S2", "text": "theta", "start": 41, "end": 46, "conf": 0.9},
    ]
    snippet, offset = _snippet_for_query(text, "theta", window=5)
    evidence_spans = _spans_for_event(spans, snippet, offset, "theta", ["S2"])
    assert evidence_spans
    for span in evidence_spans:
        assert 0 <= span.start < span.end <= len(snippet)
