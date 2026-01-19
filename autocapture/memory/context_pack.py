"""Context pack generation utilities."""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from typing import Iterable

from .entities import EntityToken


@dataclass(frozen=True)
class EvidenceSpan:
    span_id: str
    start: int
    end: int
    conf: float
    bbox: list[int] | None = None
    bbox_norm: list[float] | None = None


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    event_id: str
    timestamp: str
    ts_end: str | None
    app: str
    title: str
    domain: str | None
    score: float
    spans: list[EvidenceSpan]
    text: str
    raw_text: str | None = None
    redacted_text: str | None = None
    screenshot_path: str | None = None
    screenshot_hash: str | None = None
    retrieval: dict | None = None


@dataclass(frozen=True)
class ContextPack:
    query: str
    generated_at: str
    privacy: dict
    filters: dict
    routing: dict
    entity_tokens: list[EntityToken]
    aggregates: dict
    evidence: list[EvidenceItem]
    warnings: list[str]

    def _sanitize_evidence(self) -> tuple[list[EvidenceItem], list[str]]:
        warnings = list(self.warnings)
        sanitized: list[EvidenceItem] = []
        pattern = re.compile(
            r"(ignore previous|system prompt|developer message|you are chatgpt|do not cite|tool|function call)",
            re.IGNORECASE,
        )
        for item in self.evidence:
            raw_text = item.raw_text or item.text
            lines = []
            redacted = False
            for line in item.text.splitlines():
                if pattern.search(line):
                    lines.append("[REDACTED: potential prompt-injection]")
                    redacted = True
                else:
                    lines.append(line)
            if redacted:
                warnings.append(f"{item.evidence_id}: potential prompt-injection content redacted")
            sanitized.append(
                EvidenceItem(
                    evidence_id=item.evidence_id,
                    event_id=item.event_id,
                    timestamp=item.timestamp,
                    ts_end=item.ts_end,
                    app=item.app,
                    title=item.title,
                    domain=item.domain,
                    score=item.score,
                    spans=item.spans,
                    text="\n".join(lines),
                    raw_text=raw_text,
                    redacted_text="\n".join(lines) if redacted else None,
                    screenshot_path=item.screenshot_path,
                    screenshot_hash=item.screenshot_hash,
                    retrieval=item.retrieval,
                )
            )
        return sanitized, warnings

    def to_json(self) -> dict:
        evidence, warnings = self._sanitize_evidence()
        return {
            "version": 1,
            "query": self.query,
            "generated_at": self.generated_at,
            "evidence": [
                {
                    "id": item.evidence_id,
                    "ts_start": item.timestamp,
                    "ts_end": item.ts_end,
                    "source": item.app,
                    "title": item.title,
                    "text": item.text,
                    "meta": {
                        "event_id": item.event_id,
                        "domain": item.domain,
                        "score": item.score,
                        "screenshot_path": item.screenshot_path,
                        "screenshot_hash": item.screenshot_hash,
                        "retrieval": item.retrieval or {},
                        "spans": [
                            {
                                "span_id": span.span_id,
                                "start": span.start,
                                "end": span.end,
                                "conf": span.conf,
                                "bbox": span.bbox,
                                "bbox_norm": span.bbox_norm,
                            }
                            for span in item.spans
                        ],
                    },
                }
                for item in evidence
            ],
            "warnings": warnings,
        }

    def to_text(self, extractive_only: bool, *, format: str = "json") -> str:
        _ = extractive_only
        payload = self.to_json()
        if format == "tron":
            from ..format.tron import encode_tron

            return encode_tron(payload)
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    def to_tron(self) -> str:
        return self.to_text(extractive_only=False, format="tron")


def build_context_pack(
    query: str,
    evidence: Iterable[EvidenceItem],
    entity_tokens: list[EntityToken],
    routing: dict,
    filters: dict,
    sanitized: bool,
    aggregates: dict | None = None,
) -> ContextPack:
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return ContextPack(
        query=query,
        generated_at=generated_at,
        privacy={
            "sanitized": sanitized,
            "mode": "stable_pseudonyms",
            "notes": [],
        },
        filters=filters,
        routing=routing,
        entity_tokens=entity_tokens,
        aggregates=aggregates or {"time_spent_by_app": [], "notable_changes": []},
        evidence=list(evidence),
        warnings=[],
    )


def build_evidence_payload(context_pack: dict) -> list[dict]:
    evidence = context_pack.get("evidence") or []
    if not isinstance(evidence, list):
        return []
    payload: list[dict] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        meta = item.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        payload.append(
            {
                "id": item.get("id"),
                "event_id": meta.get("event_id"),
                "ts_start": item.get("ts_start"),
                "ts_end": item.get("ts_end"),
                "source": item.get("source"),
                "title": item.get("title"),
                "text": item.get("text"),
                "meta": meta,
            }
        )
    return payload
