"""Compression strategies for evidence into answers."""

from __future__ import annotations

from dataclasses import dataclass

from .context_pack import EvidenceItem


@dataclass(frozen=True)
class CompressedAnswer:
    answer: str
    citations: list[str]


def extractive_answer(evidence: list[EvidenceItem]) -> CompressedAnswer:
    if not evidence:
        return CompressedAnswer(
            answer="Not enough evidence to answer. Try refining the query.",
            citations=[],
        )
    lines: list[str] = []
    citations: list[str] = []
    for item in evidence:
        lines.append(f"{item.text} [{item.evidence_id}]")
        citations.append(item.evidence_id)
    return CompressedAnswer(answer="\n".join(lines), citations=citations)
