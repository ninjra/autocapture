"""Deterministic diff helpers for settings previews."""

from __future__ import annotations

from typing import Any

from .models import DiffEntry


def _is_primitive(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def diff_values(before: Any, after: Any, path: str = "") -> list[DiffEntry]:
    if before == after:
        return []

    if isinstance(before, dict) and isinstance(after, dict):
        entries: list[DiffEntry] = []
        keys = sorted(set(before.keys()) | set(after.keys()))
        for key in keys:
            next_path = f"{path}.{key}" if path else str(key)
            if key not in before:
                entries.append(DiffEntry(path=next_path, before=None, after=after[key], kind="add"))
                continue
            if key not in after:
                entries.append(
                    DiffEntry(path=next_path, before=before[key], after=None, kind="remove")
                )
                continue
            entries.extend(diff_values(before[key], after[key], next_path))
        return entries

    if isinstance(before, list) and isinstance(after, list):
        if before == after:
            return []
        return [DiffEntry(path=path, before=before, after=after, kind="change")]

    if _is_primitive(before) and _is_primitive(after):
        return [DiffEntry(path=path, before=before, after=after, kind="change")]

    return [DiffEntry(path=path, before=before, after=after, kind="change")]
