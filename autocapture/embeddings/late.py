"""Late-interaction embedding utilities (ColBERT-style)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class LateInteractionEncoder:
    def __init__(self, dim: int = 128, max_tokens: int = 64) -> None:
        self._dim = max(8, int(dim))
        self._max_tokens = max(4, int(max_tokens))

    @property
    def dim(self) -> int:
        return self._dim

    def encode_text(self, text: str) -> list[list[float]]:
        tokens = [token.lower() for token in _TOKEN_RE.findall(text or "")]
        if not tokens:
            return []
        vectors: list[list[float]] = []
        for token in tokens[: self._max_tokens]:
            vec = _hash_vector(token, self._dim)
            vectors.append(vec)
        return vectors

    def encode_many(self, texts: Iterable[str]) -> list[list[list[float]]]:
        return [self.encode_text(text) for text in texts]


def _hash_vector(token: str, dim: int) -> list[float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    values = []
    for idx in range(dim):
        byte = digest[idx % len(digest)]
        values.append((byte / 255.0) * 2.0 - 1.0)
    norm = math.sqrt(sum(val * val for val in values)) or 1.0
    return [val / norm for val in values]
