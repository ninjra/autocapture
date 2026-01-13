from __future__ import annotations

import json

import pytest

from autocapture.agents.schemas import EventEnrichmentV1
from autocapture.agents.structured_output import StructuredOutputError, parse_structured_output


def _valid_payload() -> dict:
    return {
        "schema_version": "v1",
        "event_id": "evt-1",
        "short_summary": "Working on project notes.",
        "what_i_was_doing": "Summarizing meeting notes.",
        "apps_and_tools": ["Notes"],
        "topics": ["meeting"],
        "tasks": [{"title": "Summarize notes", "status": "in_progress", "evidence": []}],
        "people": [],
        "projects": [],
        "next_actions": ["Send summary"],
        "importance": 0.5,
        "sensitivity": {"contains_pii": False, "contains_secrets": False, "notes": []},
        "keywords": ["notes"],
        "provenance": {
            "model": "test",
            "provider": "test",
            "prompt": "test",
            "created_at_utc": "2024-01-01T00:00:00+00:00",
        },
    }


def test_structured_output_parses_code_fence() -> None:
    payload = json.dumps(_valid_payload())
    text = f"```json\n{payload}\n```"
    result = parse_structured_output(text, EventEnrichmentV1)
    assert result.event_id == "evt-1"


def test_structured_output_repairs() -> None:
    bad = "not json"

    def repair(_raw: str) -> str:
        return json.dumps(_valid_payload())

    result = parse_structured_output(bad, EventEnrichmentV1, repair_fn=repair)
    assert result.short_summary


def test_structured_output_fails_without_repair() -> None:
    with pytest.raises(StructuredOutputError):
        parse_structured_output("no json here", EventEnrichmentV1)
