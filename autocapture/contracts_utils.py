"""Canonical JSON helpers for Next-10 contracts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any


def _default_encoder(value: Any) -> str:
    if isinstance(value, (dt.datetime, dt.date)):
        if isinstance(value, dt.datetime) and value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        if isinstance(value, dt.datetime):
            return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def canonical_json_dumps(payload: Any) -> str:
    """Dump JSON in canonical form (sorted keys, no whitespace)."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_default_encoder,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_canonical(payload: Any) -> str:
    return sha256_text(canonical_json_dumps(payload))


def stable_id(prefix: str, payload: Any) -> str:
    """Generate a stable content-derived ID with a prefix."""
    return f"{prefix}_{hash_canonical(payload)}"
