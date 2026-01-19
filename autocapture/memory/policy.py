"""Policy evaluation for memory artifacts and spans."""

from __future__ import annotations

import re
from typing import Iterable

from .models import ArtifactMeta, PolicyDecision


class DefaultPolicyEngine:
    def __init__(
        self,
        *,
        blocked_labels: Iterable[str],
        exclude_patterns: Iterable[str],
        redact_patterns: Iterable[str],
        redact_token: str,
    ) -> None:
        self._blocked_labels = {label.strip().lower() for label in blocked_labels if label}
        self._exclude_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in exclude_patterns if pattern
        ]
        self._redact_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in redact_patterns if pattern
        ]
        self._redact_token = redact_token or "[REDACTED]"

    def evaluate_artifact(self, meta: ArtifactMeta, text: str) -> PolicyDecision:
        labels = {label.strip().lower() for label in (meta.labels or []) if label}
        blocked = labels & self._blocked_labels
        if blocked:
            return PolicyDecision(action="exclude", reason="blocked_labels")
        if _matches_any(self._exclude_patterns, text):
            return PolicyDecision(action="exclude", reason="exclude_pattern")
        redacted_text, redaction_map = _apply_redactions(
            text, self._redact_patterns, self._redact_token
        )
        if redaction_map:
            return PolicyDecision(
                action="redact",
                redacted_text=redacted_text,
                redaction_map=redaction_map,
                reason="redact_pattern",
            )
        return PolicyDecision(action="allow")

    def evaluate_span(self, meta: ArtifactMeta, text: str) -> PolicyDecision:
        _ = meta
        if _matches_any(self._exclude_patterns, text):
            return PolicyDecision(action="exclude", reason="exclude_pattern")
        redacted_text, redaction_map = _apply_redactions(
            text, self._redact_patterns, self._redact_token
        )
        if redaction_map:
            return PolicyDecision(
                action="redact",
                redacted_text=redacted_text,
                redaction_map=redaction_map,
                reason="redact_pattern",
            )
        return PolicyDecision(action="allow")


def _matches_any(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    for pattern in patterns:
        if pattern.search(text):
            return True
    return False


def _apply_redactions(
    text: str,
    patterns: Iterable[re.Pattern[str]],
    token: str,
) -> tuple[str, list[dict[str, object]]]:
    ranges: list[tuple[int, int, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            ranges.append((match.start(), match.end(), pattern.pattern))
    if not ranges:
        return text, []
    ranges.sort(key=lambda item: (item[0], item[1], item[2]))
    merged: list[tuple[int, int, str]] = []
    for start, end, pattern in ranges:
        if not merged:
            merged.append((start, end, pattern))
            continue
        prev_start, prev_end, prev_pattern = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end), prev_pattern)
        else:
            merged.append((start, end, pattern))
    redaction_map: list[dict[str, object]] = []
    parts: list[str] = []
    cursor = 0
    for idx, (start, end, pattern) in enumerate(merged, start=1):
        parts.append(text[cursor:start])
        parts.append(f"{token}:{idx}")
        redaction_map.append({"start": start, "end": end, "pattern": pattern})
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), redaction_map
