"""Utility helpers for Memory Service."""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from typing import Iterable

from ..contracts_utils import canonical_json_dumps, sha256_text, stable_id
from ..text.normalize import normalize_text as _normalize_text


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def normalize_text(text: str) -> str:
    return _normalize_text(text or "")


def canonicalize_text(text: str) -> str:
    return normalize_text(text)


def hash_text(text: str) -> str:
    return sha256_text(text)


def hash_payload(payload: object) -> str:
    return sha256_text(canonical_json_dumps(payload))


def stable_memory_id(namespace: str, memory_type: str, content_hash: str) -> str:
    return stable_id(
        "memory",
        {"namespace": namespace, "type": memory_type, "content_hash": content_hash},
    )


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Deterministic approximation: ~4 chars per token.
    return max(1, int(math.ceil(len(text) / 4.0)))


def vector_literal(vector: Iterable[float]) -> str:
    values = [f"{float(val):.6f}" for val in vector]
    return "[" + ",".join(values) + "]"


def hash_embedding(text: str, dim: int) -> list[float]:
    if dim <= 0:
        return []
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(seed + counter.to_bytes(2, "big", signed=False)).digest()
        values.extend([byte / 255.0 for byte in digest])
        counter += 1
    return values[:dim]
