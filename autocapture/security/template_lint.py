"""Template linting utilities for prompt safety."""

from __future__ import annotations

import re

_FORBIDDEN_PATTERNS = [
    (re.compile(r"__"), "dunder sequence"),
    (re.compile(r"\|\s*attr\b"), "attr filter"),
    (re.compile(r"map\s*\(\s*attribute\s*="), "map(attribute=...)"),
    (re.compile(r"\|\s*tojson\b"), "tojson filter"),
    (re.compile(r"\|\s*safe\b"), "safe filter"),
]


def lint_template_text(template: str, *, label: str = "template") -> None:
    if not template:
        return
    for pattern, reason in _FORBIDDEN_PATTERNS:
        if pattern.search(template):
            raise ValueError(f"{label} contains forbidden pattern: {reason}")
