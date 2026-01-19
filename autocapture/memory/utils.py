"""Utility helpers for the deterministic memory store."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


EPOCH_UTC = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def stable_json_dumps(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=(",", ":") if indent is None else None,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        # Best-effort on Windows or restricted filesystems.
        pass


def normalize_document_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def normalize_title(text: str | None) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def parse_iso8601(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def format_utc(timestamp: dt.datetime | None) -> str:
    ts = timestamp or EPOCH_UTC
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def coerce_timestamp(value: str | None, fallback: dt.datetime | None = None) -> dt.datetime:
    parsed = parse_iso8601(value)
    if parsed is not None:
        return parsed
    return fallback or EPOCH_UTC


def hash_config(payload: Any) -> str:
    text = stable_json_dumps(payload, indent=None)
    return sha256_text(text)


def sanitize_fts_query(query: str) -> str:
    tokens = re.findall(r"[\w\.-]+", query or "")
    if not tokens:
        return ""
    quoted = ['"' + token.replace('"', '""') + '"' for token in tokens]
    return " AND ".join(quoted)
