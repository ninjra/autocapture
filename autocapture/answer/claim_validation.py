"""Deterministic claim + citation validation."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CitationValidatorConfig
from .claims import ClaimItem, ClaimsPayload


@dataclass(frozen=True)
class ClaimValidationResult:
    valid: bool
    errors: list[str]
    claims: list[ClaimItem]


class ClaimValidator:
    def __init__(self, config: CitationValidatorConfig) -> None:
        self._config = config

    def validate(
        self, payload: ClaimsPayload, *, valid_evidence_ids: set[str]
    ) -> ClaimValidationResult:
        errors: list[str] = []
        claims = payload.claims or []
        if not claims and not self._config.allow_empty:
            errors.append("claims_empty")
            return ClaimValidationResult(valid=False, errors=errors, claims=[])
        if len(claims) > self._config.max_claims:
            errors.append("claims_too_many")
        normalized: list[ClaimItem] = []
        for idx, claim in enumerate(claims, start=1):
            text = (claim.text or "").strip()
            if not text:
                errors.append(f"claim_{idx}_empty_text")
            evidence_ids = [cid for cid in claim.evidence_ids if cid]
            deduped = list(dict.fromkeys(evidence_ids))
            if not deduped and not self._config.allow_empty:
                errors.append(f"claim_{idx}_missing_citations")
            if len(deduped) > self._config.max_citations_per_claim:
                errors.append(f"claim_{idx}_too_many_citations")
            unknown = [cid for cid in deduped if cid not in valid_evidence_ids]
            if unknown:
                errors.append(f"claim_{idx}_unknown_citations")
            normalized.append(
                ClaimItem(
                    claim_id=claim.claim_id,
                    text=text,
                    evidence_ids=deduped,
                    entity_tokens=list(dict.fromkeys(claim.entity_tokens or [])),
                )
            )
        return ClaimValidationResult(valid=not errors, errors=errors, claims=normalized)


__all__ = ["ClaimValidator", "ClaimValidationResult"]
