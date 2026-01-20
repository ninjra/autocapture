"""Deterministic claim + citation validation."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CitationValidatorConfig
from ..memory.prompt_injection import REDACTION_MARKER
from .claims import ClaimItem, ClaimsPayload, CitationRef


@dataclass(frozen=True)
class ClaimValidationResult:
    valid: bool
    errors: list[str]
    claims: list[ClaimItem]


@dataclass(frozen=True)
class EvidenceLineInfo:
    evidence_id: str
    lines: list[str]
    citable: bool

    @property
    def line_count(self) -> int:
        return len(self.lines)


class ClaimValidator:
    def __init__(self, config: CitationValidatorConfig) -> None:
        self._config = config

    def validate(
        self,
        payload: ClaimsPayload,
        *,
        evidence_map: dict[str, EvidenceLineInfo],
    ) -> ClaimValidationResult:
        errors: list[str] = []
        claims = payload.claims or []
        valid_evidence_ids = set(evidence_map)
        abstained_ids = {
            item.claim_id for item in getattr(payload, "abstentions", []) if item.claim_id
        }
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
            citations = list(claim.citations or [])
            if not citations and claim.evidence_ids and self._config.allow_legacy_evidence_ids:
                for evidence_id in claim.evidence_ids:
                    info = evidence_map.get(evidence_id)
                    if not info:
                        continue
                    line_end = max(1, info.line_count)
                    citations.append(
                        CitationRef(
                            evidence_id=evidence_id,
                            line_start=1,
                            line_end=line_end,
                            confidence=None,
                        )
                    )
            deduped_citations: list[CitationRef] = []
            seen = set()
            for cite in citations:
                key = (cite.evidence_id, cite.line_start, cite.line_end)
                if key in seen:
                    continue
                seen.add(key)
                deduped_citations.append(cite)
            if not deduped_citations and claim.claim_id not in abstained_ids:
                if not self._config.allow_empty:
                    errors.append(f"claim_{idx}_missing_citations")
            if len(deduped_citations) > self._config.max_citations_per_claim:
                errors.append(f"claim_{idx}_too_many_citations")
            evidence_ids = [cite.evidence_id for cite in deduped_citations if cite.evidence_id]
            unknown = [cid for cid in evidence_ids if cid not in valid_evidence_ids]
            if unknown:
                errors.append(f"claim_{idx}_unknown_citations")
            for cite in deduped_citations:
                info = evidence_map.get(cite.evidence_id)
                if not info:
                    continue
                if not info.citable:
                    errors.append(f"claim_{idx}_non_citable")
                    continue
                line_count = info.line_count
                if cite.line_start < 1 or cite.line_end < 1:
                    errors.append(f"claim_{idx}_line_bounds_invalid")
                    continue
                if cite.line_start > cite.line_end:
                    errors.append(f"claim_{idx}_line_bounds_invalid")
                    continue
                if cite.line_end > line_count:
                    errors.append(f"claim_{idx}_line_out_of_bounds")
                    continue
                span = info.lines[cite.line_start - 1 : cite.line_end]
                if self._config.max_line_span and (
                    cite.line_end - cite.line_start + 1 > self._config.max_line_span
                ):
                    errors.append(f"claim_{idx}_line_span_too_large")
                if _span_redacted(span):
                    errors.append(f"claim_{idx}_span_redacted")
            normalized.append(
                ClaimItem(
                    claim_id=claim.claim_id,
                    text=text,
                    citations=deduped_citations,
                    evidence_ids=list(dict.fromkeys(evidence_ids)),
                    entity_tokens=list(dict.fromkeys(claim.entity_tokens or [])),
                )
            )
        return ClaimValidationResult(valid=not errors, errors=errors, claims=normalized)


def _span_redacted(lines: list[str]) -> bool:
    if not lines:
        return True
    for line in lines:
        if not line or not line.strip():
            continue
        if line.strip() == REDACTION_MARKER:
            continue
        if line.strip().startswith("[REDACTED"):
            continue
        return False
    return True


__all__ = ["ClaimValidator", "ClaimValidationResult", "EvidenceLineInfo"]
