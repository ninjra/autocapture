from autocapture.memory.context_pack import (
    EvidenceItem,
    EvidenceSpan,
    build_context_pack,
)
from autocapture.memory.entities import EntityToken


def test_context_pack_text_formatting() -> None:
    evidence = [
        EvidenceItem(
            evidence_id="E1",
            event_id="evt1",
            timestamp="2024-01-01T00:00:00Z",
            app="App",
            title="Window",
            domain="example.com",
            score=0.9,
            spans=[EvidenceSpan(span_id="S1", start=0, end=10, conf=0.95)],
            text="Sample text",
        )
    ]
    tokens = [EntityToken(token="ORG_1234", entity_type="ORG")]
    pack = build_context_pack(
        query="q",
        evidence=evidence,
        entity_tokens=tokens,
        routing={"llm": "local"},
        filters={"time_range": "", "apps": [], "domains": []},
        sanitized=True,
    )
    text = pack.to_text(extractive_only=True)
    assert text.startswith("===BEGIN AC_CONTEXT_PACK_V1===")
    assert text.endswith("===END AC_CONTEXT_PACK_V1===")
    assert "RULES_FOR_ASSISTANT:" in text
    assert "[E1]" in text
