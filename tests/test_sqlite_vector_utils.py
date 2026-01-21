from __future__ import annotations

from autocapture.indexing.sqlite_utils import (
    cosine_similarity,
    maxsim_score,
    vector_from_blob,
    vector_norm,
    vector_signature,
    vector_to_blob,
)


def test_vector_blob_roundtrip() -> None:
    vector = [0.1, 1.25, -2.5]
    blob = vector_to_blob(vector)
    restored = vector_from_blob(blob)
    assert len(restored) == len(vector)
    for left, right in zip(restored, vector):
        assert abs(left - right) < 1e-4


def test_vector_norm_zero_is_deterministic() -> None:
    assert vector_norm([0.0, 0.0, 0.0]) == 0.0


def test_vector_signature_is_deterministic() -> None:
    vec = [0.5, -0.25, 0.75, 0.0]
    sig1 = vector_signature(vec, seed=123, bits=16)
    sig2 = vector_signature(vec, seed=123, bits=16)
    assert sig1 == sig2


def test_cosine_similarity_zero_norm_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_maxsim_score_orders_expected() -> None:
    query = [[1.0, 0.0], [0.0, 1.0]]
    doc_good = [[1.0, 0.0], [0.0, 1.0]]
    doc_bad = [[0.5, 0.0], [0.0, 0.5]]
    assert maxsim_score(query, doc_good) > maxsim_score(query, doc_bad)
