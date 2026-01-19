from __future__ import annotations

import asyncio
import datetime as dt

from autocapture.answer.entailment import heuristic_entailment, judge_entailment
from autocapture.answer.claims import ClaimItem
from autocapture.memory.context_pack import EvidenceItem, EvidenceSpan


def _evidence(evidence_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        event_id="evt1",
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        ts_end=None,
        app="App",
        title="Title",
        domain=None,
        score=0.5,
        spans=[EvidenceSpan(span_id="s1", start=0, end=len(text), conf=0.9)],
        text=text,
        raw_text=text,
    )


def test_heuristic_entailment_numeric_mismatch() -> None:
    evidence = _evidence("E1", "Value is 100 units.")
    claim = ClaimItem(
        claim_id="c1",
        text="Value is 200 units.",
        citations=[{"evidence_id": "E1", "line_start": 1, "line_end": 1}],
    )
    result = heuristic_entailment([claim], {"E1": evidence})
    assert result.verdicts["c1"] == "not_enough_information"


def test_judge_entailment_parses_verdicts() -> None:
    class _StubProvider:
        async def generate_answer(self, *_args, **_kwargs) -> str:
            return (
                "```json\n"
                '{"schema_version":1,"verdicts":[{"claim_id":"c1","verdict":"entailed"}]}'
                "\n```"
            )

    class _StubRouter:
        def __init__(self, provider):
            self._provider = provider

        def select_llm(self, _stage: str, *, routing_override=None):
            return self._provider, object()

    evidence = _evidence("E1", "Value is 100 units.")
    claim = ClaimItem(
        claim_id="c1",
        text="Value is 100 units.",
        citations=[{"evidence_id": "E1", "line_start": 1, "line_end": 1}],
    )
    result = asyncio.run(
        judge_entailment(
            _StubRouter(_StubProvider()),
            stage="entailment_judge",
            claims=[claim],
            evidence_by_id={"E1": evidence},
        )
    )
    assert result.verdicts["c1"] == "entailed"
