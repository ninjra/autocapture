"""Deterministic vector utilities for SQLite backends."""

from __future__ import annotations

import functools
import math
import random
from array import array
from typing import Iterable, Sequence


FLOAT_ARRAY_TYPECODE = "f"


def vector_to_blob(vector: Sequence[float]) -> bytes:
    buf = array(FLOAT_ARRAY_TYPECODE, (float(val) for val in vector))
    return buf.tobytes()


def vector_from_blob(blob: bytes) -> list[float]:
    buf = array(FLOAT_ARRAY_TYPECODE)
    if blob:
        buf.frombytes(blob)
    return list(buf)


def vector_norm(vector: Sequence[float]) -> float:
    total = 0.0
    for val in vector:
        total += float(val) * float(val)
    if total <= 0.0:
        return 0.0
    return math.sqrt(total)


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
    *,
    left_norm: float | None = None,
    right_norm: float | None = None,
) -> float:
    if left_norm is None:
        left_norm = vector_norm(left)
    if right_norm is None:
        right_norm = vector_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = 0.0
    for lv, rv in zip(left, right):
        dot += float(lv) * float(rv)
    return dot / (left_norm * right_norm)


def mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    accum = [0.0] * dim
    for vec in vectors:
        for idx, val in enumerate(vec[:dim]):
            accum[idx] += float(val)
    count = float(len(vectors))
    if count == 0:
        return [0.0] * dim
    return [val / count for val in accum]


@functools.lru_cache(maxsize=16)
def _projection_matrix(seed: int, dims: int, bits: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(dims)] for _ in range(bits)]


def vector_signature(vector: Sequence[float], *, seed: int, bits: int) -> int:
    dims = len(vector)
    if dims == 0:
        return 0
    planes = _projection_matrix(seed, dims, bits)
    signature = 0
    for idx, plane in enumerate(planes):
        dot = 0.0
        for val, weight in zip(vector, plane):
            dot += float(val) * float(weight)
        if dot >= 0.0:
            signature |= 1 << idx
    return signature


def signature_bucket(signature: int, *, bits: int) -> int:
    if bits <= 0:
        return 0
    mask = (1 << bits) - 1
    return signature & mask


def maxsim_score(
    query_vectors: Sequence[Sequence[float]],
    doc_vectors: Sequence[Sequence[float]],
) -> float:
    if not query_vectors or not doc_vectors:
        return 0.0
    total = 0.0
    for qvec in query_vectors:
        q_norm = vector_norm(qvec)
        if q_norm == 0.0:
            continue
        best = 0.0
        for dvec in doc_vectors:
            dot = 0.0
            for lv, rv in zip(qvec, dvec):
                dot += float(lv) * float(rv)
            score = dot / q_norm
            if score > best:
                best = score
        total += best
    return total / float(len(query_vectors))


def dot_sparse(indices: Iterable[int], values: Iterable[float]) -> dict[int, float]:
    result: dict[int, float] = {}
    for idx, value in zip(indices, values):
        result[int(idx)] = float(value)
    return result
