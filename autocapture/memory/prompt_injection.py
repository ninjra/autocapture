"""Prompt-injection scanning and deterministic redaction."""

from __future__ import annotations

from dataclasses import dataclass
import re


REDACTION_MARKER = "[REDACTED: potential prompt-injection]"

_PATTERNS = [
    r"ignore\s+previous",
    r"system\s+prompt",
    r"developer\s+message",
    r"you\s+are\s+chatgpt",
    r"do\s+not\s+cite",
    r"tool\s+call",
    r"function\s+call",
    r"jailbreak",
    r"prompt\s+injection",
    r"override\s+instructions",
]

_PATTERN = re.compile("|".join(f"(?:{pattern})" for pattern in _PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class PromptInjectionScan:
    redacted_text: str
    risk_score: float
    redacted_lines: list[int]
    match_count: int


def scan_prompt_injection(text: str) -> PromptInjectionScan:
    if not text:
        return PromptInjectionScan(
            redacted_text="", risk_score=0.0, redacted_lines=[], match_count=0
        )
    lines = text.splitlines()
    redacted_lines: list[int] = []
    output_lines: list[str] = []
    match_count = 0
    for idx, line in enumerate(lines, start=1):
        if _PATTERN.search(line):
            output_lines.append(REDACTION_MARKER)
            redacted_lines.append(idx)
            match_count += 1
        else:
            output_lines.append(line)
    total_lines = max(len(lines), 1)
    risk_score = min(1.0, match_count / total_lines)
    return PromptInjectionScan(
        redacted_text="\n".join(output_lines),
        risk_score=risk_score,
        redacted_lines=redacted_lines,
        match_count=match_count,
    )


__all__ = ["PromptInjectionScan", "scan_prompt_injection", "REDACTION_MARKER"]
