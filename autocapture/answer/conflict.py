"""Deterministic conflict detection for evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..memory.context_pack import EvidenceItem


@dataclass(frozen=True)
class ConflictResult:
    conflict: bool
    changed_over_time: bool
    summary: dict


_NUMBER_PATTERNS = [
    (
        re.compile(r"\b(price|cost|total|amount)\b\s*[:=]?\s*\$?([0-9,.]+)", re.IGNORECASE),
        "amount",
    ),
    (re.compile(r"\b(version|ver)\b\s*[:=]?\s*([0-9.]+)", re.IGNORECASE), "version"),
    (re.compile(r"\b(status)\b\s*[:=]?\s*([a-zA-Z0-9_-]+)", re.IGNORECASE), "status"),
]


def detect_conflicts(evidence: Iterable[EvidenceItem]) -> ConflictResult:
    values_by_field: dict[str, list[tuple[str, str, str]]] = {}
    timestamps: list[str] = []
    for item in evidence:
        text = item.text or ""
        timestamps.append(item.timestamp)
        for pattern, field in _NUMBER_PATTERNS:
            for match in pattern.findall(text):
                if isinstance(match, tuple):
                    _, value = match
                else:
                    value = match
                values_by_field.setdefault(field, []).append(
                    (value, item.evidence_id, item.timestamp)
                )

    conflicts: dict[str, list[dict]] = {}
    changed = False
    for field, values in values_by_field.items():
        distinct = {}
        for value, evidence_id, ts in values:
            norm = value.replace(",", "").strip().lower()
            if norm not in distinct:
                distinct[norm] = {"value": value, "evidence_id": evidence_id, "timestamp": ts}
        if len(distinct) >= 2:
            conflicts[field] = list(distinct.values())

    if conflicts and timestamps:
        timestamps_sorted = sorted(timestamps)
        if timestamps_sorted and timestamps_sorted[0] != timestamps_sorted[-1]:
            changed = True

    summary = {
        "conflicts": conflicts,
        "changed_over_time": changed,
    }
    return ConflictResult(
        conflict=bool(conflicts) and not changed, changed_over_time=changed, summary=summary
    )
