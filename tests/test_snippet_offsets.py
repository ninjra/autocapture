from __future__ import annotations

from autocapture.api.server import _remap_spans, _snippet_for_query, _spans_for_event
from autocapture.config import DatabaseConfig
from autocapture.memory.entities import EntityResolver
from autocapture.storage.database import DatabaseManager


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


def test_sanitized_spans_remap_to_pseudonymized_text() -> None:
    text = "Email me at test@example.com for details."
    spans = [
        {"span_id": "S1", "span_key": "S1", "text": "test@example.com", "start": 12, "end": 28, "conf": 0.9},
    ]
    snippet, offset = _snippet_for_query(text, "example", window=20)
    evidence_spans = _spans_for_event(spans, snippet, offset, "example", ["S1"])
    db = DatabaseManager(DatabaseConfig(url="sqlite:///:memory:"))
    resolver = EntityResolver(db, b"secret")
    sanitized, mapping = resolver.pseudonymize_text_with_mapping(snippet)
    remapped = _remap_spans(evidence_spans, mapping, len(sanitized))
    assert remapped
    for span in remapped:
        assert 0 <= span.start < span.end <= len(sanitized)
        assert sanitized[span.start : span.end].startswith("EMAIL_")
