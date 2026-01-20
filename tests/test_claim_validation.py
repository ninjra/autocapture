from __future__ import annotations

from autocapture.answer.claim_validation import ClaimValidator, EvidenceLineInfo
from autocapture.answer.claims import ClaimsPayload, parse_claims_json
from autocapture.config import CitationValidatorConfig


def test_parse_claims_assigns_ids() -> None:
    text = (
        "```json\n"
        '{"schema_version":2,"claims":[{"text":"Claim A","citations":[{"evidence_id":"E1","line_start":1,"line_end":1}],"entity_tokens":[]}]}'
        "\n```"
    )
    parsed = parse_claims_json(text)
    assert parsed.payload.claims
    assert parsed.payload.claims[0].claim_id


def test_claim_validator_rejects_unknown_citations() -> None:
    payload = ClaimsPayload(
        schema_version=2,
        claims=[
            {
                "claim_id": "c1",
                "text": "Claim",
                "citations": [{"evidence_id": "E9", "line_start": 1, "line_end": 1}],
                "entity_tokens": [],
            }
        ],
    )
    validator = ClaimValidator(
        CitationValidatorConfig(max_claims=10, max_citations_per_claim=3, allow_empty=False)
    )
    evidence_map = {"E1": EvidenceLineInfo(evidence_id="E1", lines=["Evidence"], citable=True)}
    result = validator.validate(payload, evidence_map=evidence_map)
    assert not result.valid
