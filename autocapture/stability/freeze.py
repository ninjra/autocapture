"""Frozen surface tooling for stability enforcement."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

MANIFEST_RELATIVE_PATH = Path("autocapture/stability/frozen_manifest.json")
SCHEMA_VERSION = 1

logger = logging.getLogger("autocapture.stability.freeze")


def _manifest_path(repo_root: Path) -> Path:
    return repo_root / MANIFEST_RELATIVE_PATH


def _empty_manifest() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "frozen": {}}


def _ensure_manifest_schema(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object.")
    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported manifest schema version. "
            f"Expected {SCHEMA_VERSION}, got {schema_version!r}."
        )
    frozen = manifest.get("frozen")
    if frozen is None:
        manifest["frozen"] = {}
    elif not isinstance(frozen, dict):
        raise ValueError("Manifest 'frozen' field must be an object.")
    return manifest


def load_manifest(repo_root: Path) -> dict[str, Any]:
    """Load the frozen manifest from disk."""
    manifest_path = _manifest_path(repo_root)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Frozen manifest not found at {manifest_path}. "
            "Run freeze_surfaces.py to initialize it."
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Frozen manifest contains invalid JSON.") from exc
    return _ensure_manifest_schema(data)


def save_manifest(repo_root: Path, manifest: dict[str, Any]) -> None:
    """Persist the manifest with an atomic write."""
    manifest_path = _manifest_path(repo_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(_ensure_manifest_schema(manifest), indent=2, sort_keys=True)
    temp_path = manifest_path.with_suffix(".tmp")
    temp_path.write_text(f"{data}\n", encoding="utf-8")
    temp_path.replace(manifest_path)


def sha256_file(path: Path) -> str:
    """Return the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def freeze_files(repo_root: Path, rel_paths: list[str], reason: str) -> None:
    """Freeze files by recording their hashes in the manifest."""
    if not reason.strip():
        raise ValueError("Freeze reason must be a non-empty string.")
    manifest = load_manifest(repo_root)
    frozen = manifest.setdefault("frozen", {})
    for rel_path in rel_paths:
        path = repo_root / rel_path
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Cannot freeze missing file: {rel_path}")
        frozen[rel_path] = {
            "sha256": sha256_file(path),
            "frozen_at_utc": _utc_timestamp(),
            "reason": reason,
        }
        logger.info("Froze %s", rel_path)
    save_manifest(repo_root, manifest)


def unfreeze_files(repo_root: Path, rel_paths: list[str], reason: str) -> None:
    """Remove files from the frozen manifest."""
    if not reason.strip():
        raise ValueError("Unfreeze reason must be a non-empty string.")
    manifest = load_manifest(repo_root)
    frozen = manifest.setdefault("frozen", {})
    missing: list[str] = []
    for rel_path in rel_paths:
        if rel_path not in frozen:
            missing.append(rel_path)
            continue
        frozen.pop(rel_path, None)
        logger.info("Unfroze %s", rel_path)
    if missing:
        raise ValueError(
            "Cannot unfreeze paths not present in manifest: " + ", ".join(missing)
        )
    save_manifest(repo_root, manifest)


def verify_frozen(repo_root: Path) -> list[str]:
    """Return a list of frozen file violations."""
    manifest = load_manifest(repo_root)
    frozen = manifest.get("frozen", {})
    violations: list[str] = []
    for rel_path, metadata in frozen.items():
        expected = metadata.get("sha256")
        path = repo_root / rel_path
        if not path.exists() or not path.is_file():
            message = f"{rel_path}: expected {expected}, actual missing"
            violations.append(message)
            # NOTE: stdlib logging does not support structlog-style brace formatting
            # nor arbitrary keyword args (e.g., message=...). Use %s formatting.
            logger.error("Frozen surface violation: %s", message)
            continue
        actual = sha256_file(path)
        if actual != expected:
            message = f"{rel_path}: expected {expected}, actual {actual}"
            violations.append(message)
            logger.error("Frozen surface violation: %s", message)
    return violations
