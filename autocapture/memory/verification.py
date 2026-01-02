"""Deterministic verifier for evidence-linked claims."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    text: str
    evidence_ids: list[str]
    entity_tokens: list[str]


class RulesVerifier:
    def verify(self, claims: list[Claim], valid_evidence: set[str], entity_tokens: set[str]) -> list[str]:
        errors: list[str] = []
        for idx, claim in enumerate(claims, start=1):
            if not claim.evidence_ids:
                errors.append(f"Claim {idx} missing evidence IDs")
                continue
            if not set(claim.evidence_ids).issubset(valid_evidence):
                errors.append(f"Claim {idx} references unknown evidence")
            if not set(claim.entity_tokens).issubset(entity_tokens):
                errors.append(f"Claim {idx} references unknown entity tokens")
        return errors
