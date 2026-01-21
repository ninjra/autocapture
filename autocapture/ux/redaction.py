"""UX redaction helpers (wrapper around core redaction utilities)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..security.redaction import redact_mapping, redact_text, redact_value


def redact_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return redact_mapping(payload)
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    if isinstance(payload, str):
        return redact_text(payload)
    if isinstance(payload, Path):
        return str(payload)
    return redact_value(payload)


def normalize_path(path: str | Path, *, root: Path | None = None) -> str:
    raw = Path(path)
    if root:
        try:
            return str(raw.resolve().relative_to(root.resolve()))
        except Exception:
            return str(raw)
    return str(raw)
