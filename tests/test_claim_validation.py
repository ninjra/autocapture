from __future__ import annotations

from autocapture.answer.claim_validation import ClaimValidator
from autocapture.answer.claims import ClaimsPayload, parse_claims_json
from autocapture.config import CitationValidatorConfig


def test_parse_claims_assigns_ids() -> None:
    text = (
        "```json\n"
        '{"schema_version":1,"claims":[{"text":"Claim A","evidence_ids":["E1"],"entity_tokens":[]}]}'
        "\n```"
    )
    parsed = parse_claims_json(text)
    assert parsed.payload.claims
    assert parsed.payload.claims[0].claim_id


def test_claim_validator_rejects_unknown_citations() -> None:
    payload = ClaimsPayload(
        schema_version=1,
        claims=[
            {
                "claim_id": "c1",
                "text": "Claim",
                "evidence_ids": ["E9"],
                "entity_tokens": [],
            }
        ],
    )
    validator = ClaimValidator(
        CitationValidatorConfig(max_claims=10, max_citations_per_claim=3, allow_empty=False)
    )
    result = validator.validate(payload, valid_evidence_ids={"E1"})
    assert not result.valid
