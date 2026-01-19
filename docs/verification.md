# Verification (claims + entailment)

Autocapture can enforce structured, claim-level answers with deterministic validation and
an entailment gate.

## claims_json_v1 schema
```json
{
  "schema_version": 1,
  "claims": [
    {
      "claim_id": "optional-stable-id",
      "text": "Claim text",
      "evidence_ids": ["E1", "E2"],
      "entity_tokens": []
    }
  ],
  "answer": "optional_summary"
}
```

## Validation
The validator enforces:
- non-empty claim text
- citations limited to known evidence IDs
- per-claim citation caps

Validation settings live under `verification.citation_validator`.

## Entailment gate
When enabled, the entailment gate applies:
- a deterministic heuristic check (numbers/entities)
- an LLM judge stage (`entailment_judge`)

Policies are configured under `verification.entailment`:
- `on_contradiction`: `block` or `regenerate`
- `on_nei`: `abstain` or `expand_retrieval`
