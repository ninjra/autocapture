"""Checksum-verified native extension loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import AppConfig
from ..logging_utils import get_logger


_LOG = get_logger("security.extensions")


def load_checksums(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        return {}
    result: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, dict):
            continue
        key = entry.get("path")
        value = entry.get("sha256")
        if isinstance(key, str) and isinstance(value, str):
            result[key] = value
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup_allowlist_entry(path: Path, allowlist: dict[str, str]) -> tuple[str | None, str | None]:
    candidates: list[str] = [path.as_posix()]
    try:
        candidates.append(path.resolve().as_posix())
    except Exception:
        pass
    try:
        candidates.append(path.relative_to(Path.cwd()).as_posix())
    except Exception:
        pass
    candidates.append(path.name)
    for key in candidates:
        expected = allowlist.get(key)
        if expected is not None:
            return key, expected
    return None, None


def verify_checksum(path: Path, allowlist: dict[str, str]) -> tuple[bool, str]:
    _key, expected = _lookup_allowlist_entry(path, allowlist)
    if expected is None:
        return False, "checksum_missing"
    if not path.exists():
        return False, "file_missing"
    actual = sha256_file(path)
    if actual != expected:
        return False, "checksum_mismatch"
    return True, "ok"


def guarded_load_extension(conn, extension_path: Path, *, config: AppConfig) -> bool:
    policy_path = Path("CHECKSUMS.json")
    allowlist = load_checksums(policy_path)
    ok, reason = verify_checksum(extension_path, allowlist)
    if not ok:
        message = f"Extension checksum verification failed: {extension_path} ({reason})"
        if config.security.secure_mode:
            raise RuntimeError(message)
        _LOG.warning(message)
        return False
    conn.enable_load_extension(True)
    try:
        conn.load_extension(str(extension_path))
    finally:
        conn.enable_load_extension(False)
    return True
