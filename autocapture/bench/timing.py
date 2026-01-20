"""Lightweight JSONL timing tracer for bench runs."""

from __future__ import annotations

import json
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..security.redaction import redact_text

_SAFE_STRING_FIELDS = {
    "case_id",
    "mode",
    "provider",
    "status",
    "format",
}


@dataclass
class TimingTracer:
    enabled: bool
    redact: bool = True
    run_id: str | None = None
    file_path: Path | None = None

    def __post_init__(self) -> None:
        self._run_id = self.run_id or uuid.uuid4().hex
        self._handle = None
        self._owns_handle = False
        if self.enabled:
            if self.file_path:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self.file_path.open("a", encoding="utf-8")
                self._owns_handle = True
            else:
                self._handle = sys.stderr

    def close(self) -> None:
        if self._owns_handle and self._handle:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "TimingTracer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    @contextmanager
    def span(self, phase: str, **fields: Any) -> Iterable[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(phase, elapsed_ms, fields)

    def record(self, phase: str, elapsed_ms: float, fields: dict[str, Any] | None = None) -> None:
        if not self.enabled or self._handle is None:
            return
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self._run_id,
            "phase": phase,
            "ms": round(float(elapsed_ms), 4),
            "fields": self._sanitize_fields(fields or {}),
        }
        self._handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._handle.flush()

    def _sanitize_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        if not self.redact:
            return fields
        sanitized: dict[str, Any] = {}
        for key, value in fields.items():
            if isinstance(value, (int, float, bool)):
                sanitized[key] = value
                continue
            if isinstance(value, str):
                if key in _SAFE_STRING_FIELDS:
                    sanitized[key] = redact_text(value)
                else:
                    sanitized[key] = "[REDACTED]"
                continue
            if isinstance(value, dict):
                sanitized[key] = {"keys": sorted(value.keys())}
                continue
            if isinstance(value, list):
                sanitized[key] = {"len": len(value)}
                continue
            sanitized[key] = "[REDACTED]"
        return sanitized
