"""Perf log reader for dashboard visibility."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_perf_log(
    data_dir: Path,
    *,
    component: str,
    limit: int = 200,
    max_bytes: int = 1_000_000,
) -> dict[str, Any]:
    component = component.strip().lower()
    if component not in {"runtime", "api"}:
        raise ValueError("component must be 'runtime' or 'api'")
    path = Path(data_dir) / "perf" / f"{component}.jsonl"
    lines = _tail_lines(path, limit=limit, max_bytes=max_bytes)
    entries: list[dict[str, Any]] = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            entries.append({"raw": raw, "parsed": parsed})
        except Exception:
            entries.append({"raw": raw, "parsed": None})
    return {
        "component": component,
        "path": str(path),
        "entries": entries,
    }


def _tail_lines(path: Path, *, limit: int, max_bytes: int) -> list[str]:
    if limit <= 0:
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size <= 0:
        return []
    start = max(0, size - max_bytes)
    try:
        with path.open("rb") as handle:
            handle.seek(start, os.SEEK_SET)
            data = handle.read()
    except OSError:
        return []
    lines = data.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    if limit and len(lines) > limit:
        lines = lines[-limit:]
    return [line.decode("utf-8", errors="replace") for line in lines]
