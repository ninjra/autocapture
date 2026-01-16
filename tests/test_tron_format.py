from __future__ import annotations

from pydantic import BaseModel

from autocapture.agents.structured_output import parse_structured_output
from autocapture.format.tron import decode_tron, encode_tron


class _RefinedQuery(BaseModel):
    refined_query: str


def test_tron_encode_decode_roundtrip() -> None:
    data = {
        "version": 1,
        "query": "q",
        "generated_at": "2026-01-15T00:00:00Z",
        "evidence": [
            {
                "id": "E1",
                "ts_start": "2026-01-15T00:00:00Z",
                "ts_end": None,
                "source": "app",
                "title": "title",
                "text": "hello",
                "meta": {
                    "event_id": "evt1",
                    "domain": None,
                    "score": 0.5,
                    "spans": [{"span_id": "S1", "start": 0, "end": 5, "conf": 0.9}],
                },
            }
        ],
        "warnings": [],
    }
    tron = encode_tron(data)
    assert "class EvidenceItem" in tron
    decoded = decode_tron(tron)
    assert decoded["evidence"][0]["id"] == "E1"
    assert decoded["evidence"][0]["meta"]["spans"][0]["span_id"] == "S1"


def test_tron_decode_accepts_json() -> None:
    payload = decode_tron('{"ok": true, "value": 2}')
    assert payload == {"ok": True, "value": 2}


def test_structured_output_accepts_tron() -> None:
    tron = 'class Query { refined_query }\n\nQuery("hello")'
    parsed = parse_structured_output(tron, _RefinedQuery)
    assert parsed.refined_query == "hello"
