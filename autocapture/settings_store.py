"""Persistence helpers for settings.json."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable

from .fs_utils import fsync_dir, fsync_file, safe_replace, safe_unlink


def read_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def write_settings(path: Path, settings: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp-{path.name}-{uuid.uuid4().hex}")
    try:
        tmp_path.write_text(_safe_json(settings), encoding="utf-8")
        fsync_file(tmp_path)
        fsync_dir(tmp_path.parent)
        safe_replace(tmp_path, path)
        fsync_dir(path.parent)
    finally:
        if tmp_path.exists():
            safe_unlink(tmp_path)


def update_settings(
    path: Path, updater: Callable[[dict[str, Any]], dict[str, Any] | None]
) -> dict[str, Any]:
    current = read_settings(path)
    updated = updater(current)
    if updated is None:
        updated = current
    write_settings(path, updated)
    return updated


def _safe_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
