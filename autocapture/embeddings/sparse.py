"""Sparse embedding utilities (SPLADE-like)."""

from __future__ import annotations

import importlib.util
import math
import re
import hashlib
from typing import Iterable

from ..logging_utils import get_logger
from ..indexing.spans_v2 import SparseEmbedding


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class SparseEncoder:
    def __init__(self, model_name: str = "hash-splade") -> None:
        self._model_name = model_name
        self._backend = None
        self._log = get_logger("sparse_embed")
        self._init_backend()

    def _init_backend(self) -> None:
        if self._model_name in {"hash", "hash-splade", "local-test"}:
            self._backend = "hash"
            return
        if importlib.util.find_spec("fastembed") is not None:
            try:
                from fastembed import SparseTextEmbedding  # type: ignore

                self._backend = SparseTextEmbedding(model_name=self._model_name)
                self._log.info("Sparse backend: fastembed ({})", self._model_name)
                return
            except Exception as exc:
                self._log.warning("Sparse backend init failed: {}", exc)
        self._backend = "hash"
        self._log.warning("Sparse backend fallback to hash for {}", self._model_name)

    def encode(self, texts: Iterable[str]) -> list[SparseEmbedding]:
        items = [text or "" for text in texts]
        if not items:
            return []
        if self._backend == "hash":
            return [_hash_sparse(text) for text in items]
        if self._backend is None:
            self._init_backend()
        if self._backend is None or self._backend == "hash":
            return [_hash_sparse(text) for text in items]
        vectors = list(self._backend.embed(items))
        embeddings: list[SparseEmbedding] = []
        for vector in vectors:
            indices = list(getattr(vector, "indices", []))
            values = list(getattr(vector, "values", []))
            embeddings.append(SparseEmbedding(indices=indices, values=values))
        return embeddings


def _hash_sparse(text: str) -> SparseEmbedding:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text or "")]
    counts: dict[int, int] = {}
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little", signed=False)
        counts[idx] = counts.get(idx, 0) + 1
    indices: list[int] = []
    values: list[float] = []
    for idx in sorted(counts):
        indices.append(idx)
        values.append(float(math.log1p(counts[idx])))
    return SparseEmbedding(indices=indices, values=values)
