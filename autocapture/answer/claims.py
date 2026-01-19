"""Claim-level JSON schema helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel, Field, ValidationError

from ..contracts_utils import stable_id


class ClaimItem(BaseModel):
    claim_id: str | None = None
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    entity_tokens: list[str] = Field(default_factory=list)


class ClaimsPayload(BaseModel):
    schema_version: int = Field(1, ge=1)
    claims: list[ClaimItem] = Field(default_factory=list)
    answer: str | None = None
    entailment: dict[str, str] | None = None


@dataclass(frozen=True)
class ParsedClaims:
    payload: ClaimsPayload
    raw_json: dict


def _extract_json_block(text: str) -> str:
    if not text:
        raise ValueError("empty response")
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object found")
    return text[start : end + 1].strip()


def parse_claims_json(text: str) -> ParsedClaims:
    raw_text = _extract_json_block(text)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid json") from exc
    try:
        model = ClaimsPayload.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("invalid claims schema") from exc
    _ensure_claim_ids(model)
    return ParsedClaims(payload=model, raw_json=payload)


def _ensure_claim_ids(payload: ClaimsPayload) -> None:
    for claim in payload.claims:
        if claim.claim_id:
            continue
        claim.claim_id = stable_id(
            "claim",
            {
                "text": claim.text,
                "evidence_ids": claim.evidence_ids,
                "entity_tokens": claim.entity_tokens,
            },
        )


def render_claims_answer(claims: Iterable[ClaimItem]) -> str:
    lines: list[str] = []
    for claim in claims:
        text = (claim.text or "").strip()
        if not text:
            continue
        citations = " ".join(f"[{cite}]" for cite in claim.evidence_ids)
        lines.append(f"{text} {citations}".strip())
    return "\n".join(lines).strip()


__all__ = ["ClaimItem", "ClaimsPayload", "ParsedClaims", "parse_claims_json", "render_claims_answer"]
