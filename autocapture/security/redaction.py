"""Secret and sensitive data redaction helpers."""

from __future__ import annotations

import re
from typing import Any

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)bearer\\s+[A-Za-z0-9._+/=-]{8,}"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)api[_-]?key\\s*[:=]\\s*[A-Za-z0-9._+/=-]{6,}"), "[REDACTED]"),
    (re.compile(r"(?i)token\\s*[:=]\\s*[A-Za-z0-9._+/=-]{6,}"), "[REDACTED]"),
    (re.compile(r"(?i)secret\\s*[:=]\\s*[A-Za-z0-9._+/=-]{6,}"), "[REDACTED]"),
    (re.compile(r"(?i)sk-[A-Za-z0-9-]{8,}"), "sk-[REDACTED]"),
    (re.compile(r"(?i)gh[pous]_[A-Za-z0-9-]{8,}"), "gh[REDACTED]"),
]

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "client_secret",
    "access_token",
    "refresh_token",
    "authorization",
    "ocr_text",
    "window_title",
    "query",
    "url",
    "last_window_title_raw",
    "last_browser_url_raw",
    "raw_window_title",
    "raw_browser_url",
    "display_name",
    "normalized_title",
}


def redact_text(text: str) -> str:
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_value(value: Any, *, key: str | None = None) -> Any:
    if key and key.lower() in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return redact_mapping(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


def redact_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: redact_value(value, key=key) for key, value in payload.items()}
