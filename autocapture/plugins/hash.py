"""Hash utilities for plugins."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

try:
    from importlib.abc import Traversable
except ImportError:  # pragma: no cover - py<3.11
    from importlib.resources.abc import Traversable  # type: ignore


_EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}

_EXCLUDE_FILES = {".DS_Store"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_directory(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(_iter_files(root)):
        rel = str(path.relative_to(root)).replace("\\", "/")
        digest.update(rel.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hash_traversable(root: Traversable) -> str:
    digest = hashlib.sha256()
    items: list[tuple[str, Traversable]] = []
    _collect_traversable(root, "", items)
    for rel, entry in sorted(items, key=lambda item: item[0]):
        digest.update(rel.encode("utf-8"))
        digest.update(entry.read_bytes())
    return digest.hexdigest()


def hash_distribution_files(dist) -> str:
    digest = hashlib.sha256()
    files = list(dist.files or [])
    for entry in sorted(files, key=lambda item: str(item)):
        parts = str(entry).split("/")
        if any(part in _EXCLUDE_DIRS for part in parts):
            continue
        if ".dist-info" in parts:
            continue
        if parts and parts[-1] in _EXCLUDE_FILES:
            continue
        if parts and parts[-1].endswith(".pyc"):
            continue
        try:
            path = dist.locate_file(entry)
            if path.is_dir():
                continue
            rel = str(entry).replace("\\", "/")
            digest.update(rel.encode("utf-8"))
            digest.update(path.read_bytes())
        except Exception:
            continue
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in _EXCLUDE_DIRS:
                continue
            continue
        if path.name in _EXCLUDE_FILES or path.name.endswith(".pyc"):
            continue
        parts = path.parts
        if any(part in _EXCLUDE_DIRS for part in parts):
            continue
        yield path


def _collect_traversable(
    root: Traversable, prefix: str, items: list[tuple[str, Traversable]]
) -> None:
    for entry in root.iterdir():
        name = entry.name
        if name in _EXCLUDE_DIRS or name in _EXCLUDE_FILES or name.endswith(".pyc"):
            continue
        rel = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"
        if entry.is_dir():
            _collect_traversable(entry, rel, items)
        else:
            items.append((rel, entry))
