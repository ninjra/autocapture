"""Entailment verification (heuristic + LLM judge)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel, Field, ValidationError

from ..logging_utils import get_logger
from ..model_ops import StageRouter
from ..memory.context_pack import EvidenceItem
from .claims import ClaimItem


class EntailmentJudgement(BaseModel):
    claim_id: str
    verdict: str = Field(..., description="entailed|contradicted|nei")
    rationale: str | None = None


class EntailmentResponse(BaseModel):
    schema_version: int = Field(1, ge=1)
    verdicts: list[EntailmentJudgement] = Field(default_factory=list)


@dataclass(frozen=True)
class EntailmentResult:
    verdicts: dict[str, str]
    rationale: dict[str, str]


_LOG = get_logger("entailment")


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


def _extract_numbers(text: str) -> set[str]:
    return {match.group(0) for match in re.finditer(r"\d+(?:\.\d+)?", text or "")}


def heuristic_entailment(
    claims: Iterable[ClaimItem], evidence_by_id: dict[str, EvidenceItem]
) -> EntailmentResult:
    verdicts: dict[str, str] = {}
    rationale: dict[str, str] = {}
    for claim in claims:
        text = claim.text or ""
        numbers = _extract_numbers(text)
        evidence_text = "\n".join(
            (evidence_by_id[cid].raw_text or evidence_by_id[cid].text)
            for cid in claim.evidence_ids
            if cid in evidence_by_id
        )
        if numbers:
            evidence_numbers = _extract_numbers(evidence_text)
            if not numbers.issubset(evidence_numbers):
                verdicts[claim.claim_id or ""] = "nei"
                rationale[claim.claim_id or ""] = "numeric_mismatch"
                continue
        verdicts[claim.claim_id or ""] = "entailed"
        rationale[claim.claim_id or ""] = "heuristic_pass"
    return EntailmentResult(verdicts=verdicts, rationale=rationale)


def _build_prompt(claims: list[ClaimItem], evidence_by_id: dict[str, EvidenceItem]) -> str:
    payload = {
        "schema_version": 1,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "evidence": [
                    {
                        "evidence_id": evidence_id,
                        "text": (
                            evidence_by_id[evidence_id].raw_text or evidence_by_id[evidence_id].text
                        ),
                    }
                    for evidence_id in claim.evidence_ids
                    if evidence_id in evidence_by_id
                ],
            }
            for claim in claims
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_response(text: str) -> EntailmentResponse:
    raw_text = _extract_json_block(text)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid json") from exc
    try:
        return EntailmentResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("invalid entailment schema") from exc


async def judge_entailment(
    stage_router: StageRouter,
    *,
    stage: str,
    claims: list[ClaimItem],
    evidence_by_id: dict[str, EvidenceItem],
) -> EntailmentResult:
    system_prompt = (
        "You are a strict entailment judge. "
        "Given claims and their cited evidence, decide if each claim is "
        "entailed, contradicted, or not enough information (nei). "
        "Return JSON only with schema_version=1 and verdicts[]."
    )
    user_prompt = _build_prompt(claims, evidence_by_id)
    try:
        provider, _decision = stage_router.select_llm(stage)
        response = await provider.generate_answer(
            system_prompt,
            user_prompt,
            "",
            temperature=0.0,
        )
        parsed = _parse_response(response)
    except Exception as exc:
        _LOG.warning("Entailment judge failed: {}", exc)
        return EntailmentResult(verdicts={}, rationale={})
    verdicts: dict[str, str] = {}
    rationale: dict[str, str] = {}
    for item in parsed.verdicts:
        verdicts[item.claim_id] = item.verdict.lower()
        if item.rationale:
            rationale[item.claim_id] = item.rationale
    return EntailmentResult(verdicts=verdicts, rationale=rationale)


__all__ = ["EntailmentResult", "heuristic_entailment", "judge_entailment"]
