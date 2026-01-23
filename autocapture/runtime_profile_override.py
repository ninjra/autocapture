"""File-based runtime profile overrides shared across processes."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Tuple

from .fs_utils import safe_replace, safe_unlink
from .runtime_env import ProfileName


PROFILE_OVERRIDE_SCHEMA_VERSION = 1


def read_profile_override(path: Path) -> Tuple[bool, ProfileName | None]:
    if not path.exists():
        return False, None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False, None
    if not isinstance(raw, dict):
        return False, None
    profile_raw = str(raw.get("profile") or "").strip().lower()
    if profile_raw in {"", "auto", "balanced", "default"}:
        return True, None
    if profile_raw == ProfileName.FOREGROUND.value:
        return True, ProfileName.FOREGROUND
    if profile_raw == ProfileName.IDLE.value:
        return True, ProfileName.IDLE
    return False, None


def write_profile_override(
    path: Path,
    profile: ProfileName | None,
    *,
    source: str | None = None,
) -> None:
    payload = {
        "schema_version": PROFILE_OVERRIDE_SCHEMA_VERSION,
        "profile": profile.value if profile else "auto",
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp-{path.name}-{uuid.uuid4().hex}")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        safe_replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            safe_unlink(tmp_path)
