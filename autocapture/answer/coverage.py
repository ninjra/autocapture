"""Coverage scoring and deterministic sentence splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..contracts_utils import hash_canonical

_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "vs.",
    "prof.",
    "sr.",
    "jr.",
}


@dataclass(frozen=True)
class SentenceCoverage:
    sentence_id: str
    index: int
    text: str
    citations: list[str]


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    sentences: list[str] = []
    buffer: list[str] = []
    tokens = re.split(r"(\s+)", text)
    for token in tokens:
        buffer.append(token)
        if token.endswith((".", "?", "!")):
            segment = "".join(buffer).strip()
            if not segment:
                buffer = []
                continue
            lowered = segment.lower()
            if any(lowered.endswith(abbrev) for abbrev in _ABBREVIATIONS):
                continue
            sentences.append(segment)
            buffer = []
    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)
    return sentences


def sentence_id(text: str, index: int) -> str:
    return f"S{index:03d}_{hash_canonical({'text': text, 'index': index})[:10]}"


def extract_sentence_citations(text: str, evidence_ids: set[str]) -> list[SentenceCoverage]:
    sentences = split_sentences(text)
    coverage: list[SentenceCoverage] = []
    citation_pattern = re.compile(r"(?:\[|【)(E\d+)(?::L\d+-L\d+)?(?:\]|】)")
    for idx, sentence in enumerate(sentences, start=1):
        found = [c for c in citation_pattern.findall(sentence) if c in evidence_ids]
        coverage.append(
            SentenceCoverage(
                sentence_id=sentence_id(sentence, idx),
                index=idx,
                text=sentence,
                citations=list(dict.fromkeys(found)),
            )
        )

    def _is_citation_only(sentence: str) -> bool:
        stripped = sentence.strip()
        if not stripped:
            return False
        cleaned = citation_pattern.sub("", stripped).strip()
        return cleaned == ""

    def _leading_citations(sentence: str) -> list[str]:
        match = re.match(r"^\s*((?:\[|【)(?:E\d+)(?::L\d+-L\d+)?(?:\]|】)\s*)+", sentence)
        if not match:
            return []
        return [c for c in citation_pattern.findall(match.group(0)) if c in evidence_ids]

    for idx in range(1, len(coverage)):
        lead = _leading_citations(coverage[idx].text)
        if lead and not coverage[idx - 1].citations:
            combined = list(dict.fromkeys(coverage[idx - 1].citations + lead))
            coverage[idx - 1] = SentenceCoverage(
                sentence_id=coverage[idx - 1].sentence_id,
                index=coverage[idx - 1].index,
                text=coverage[idx - 1].text,
                citations=combined,
            )
            remaining = [c for c in coverage[idx].citations if c not in lead]
            coverage[idx] = SentenceCoverage(
                sentence_id=coverage[idx].sentence_id,
                index=coverage[idx].index,
                text=coverage[idx].text,
                citations=remaining,
            )

    merged: list[SentenceCoverage] = []
    for entry in coverage:
        if _is_citation_only(entry.text) and merged:
            combined = list(dict.fromkeys(merged[-1].citations + entry.citations))
            merged[-1] = SentenceCoverage(
                sentence_id=merged[-1].sentence_id,
                index=merged[-1].index,
                text=merged[-1].text,
                citations=combined,
            )
            continue
        merged.append(entry)
    return merged


def is_meta_sentence(text: str) -> bool:
    lowered = text.lower()
    return "no evidence" in lowered or "not enough evidence" in lowered


def coverage_metrics(
    answer_text: str,
    evidence_ids: set[str],
    *,
    no_evidence_mode: bool,
) -> dict:
    sentences = extract_sentence_citations(answer_text, evidence_ids)
    if no_evidence_mode:
        sentences = [s for s in sentences if not is_meta_sentence(s.text)]
    total = len(sentences)
    cited = sum(1 for s in sentences if s.citations)
    sentence_coverage = cited / total if total else 0.0
    uncited = [
        {"sentence_id": s.sentence_id, "index": s.index, "text": s.text}
        for s in sentences
        if not s.citations
    ]
    return {
        "sentence_coverage": sentence_coverage,
        "sentences": [
            {
                "sentence_id": s.sentence_id,
                "index": s.index,
                "text": s.text,
                "citations": s.citations,
            }
            for s in sentences
        ],
        "uncited_sentences": uncited,
    }
