"""Spans v2 vector index with dense+sparse+late vectors."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from ..resilience import CircuitBreaker, RetryPolicy, is_retryable_exception, retry_sync
from .vector_index import VectorHit, IndexUnavailable


@dataclass(frozen=True)
class SparseEmbedding:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class SpanV2Upsert:
    capture_id: str
    span_key: str
    dense_vector: list[float]
    sparse_vector: SparseEmbedding | None
    late_vectors: list[list[float]] | None
    payload: dict
    embedding_model: str


class SpansV2Index:
    def __init__(self, config: AppConfig, dim: int, *, backend: Optional[object] = None) -> None:
        self._config = config
        self._dim = dim
        self._log = get_logger("index.spans_v2")
        self._backend = backend
        if self._backend is None and config.qdrant.enabled:
            self._backend = QdrantSpansV2Backend(config, dim)
            self._log.info("Spans v2 index: Qdrant")
        elif self._backend is None:
            self._log.info("Spans v2 index: disabled")

    def upsert(self, upserts: list[SpanV2Upsert]) -> None:
        if not self._backend:
            return
        try:
            self._backend.upsert(upserts)
        except IndexUnavailable:
            raise
        except Exception as exc:
            self._log.warning("Spans v2 upsert failed: {}", exc)

    def search_dense(
        self, vector: list[float], k: int, *, filters: dict | None = None, embedding_model: str
    ) -> list[VectorHit]:
        if not self._backend:
            return []
        return self._backend.search_dense(
            vector, k, filters=filters, embedding_model=embedding_model
        )

    def search_sparse(
        self, vector: SparseEmbedding, k: int, *, filters: dict | None = None
    ) -> list[VectorHit]:
        if not self._backend:
            return []
        return self._backend.search_sparse(vector, k, filters=filters)

    def search_late(
        self, vectors: list[list[float]], k: int, *, filters: dict | None = None
    ) -> list[VectorHit]:
        if not self._backend:
            return []
        return self._backend.search_late(vectors, k, filters=filters)


class QdrantSpansV2Backend:
    def __init__(self, config: AppConfig, dim: int) -> None:
        from qdrant_client import QdrantClient

        self._config = config
        self._dim = dim
        self._collection = config.qdrant.spans_v2_collection
        self._client = QdrantClient(url=config.qdrant.url, timeout=2.0)
        self._log = get_logger("index.qdrant.spans_v2")
        self._retry_policy = RetryPolicy()
        self._breaker = CircuitBreaker()
        self._ready = False
        self._lock = threading.Lock()

    def _run(self, fn):
        if not self._breaker.allow():
            raise IndexUnavailable("spans_v2 index unavailable (circuit open)")
        try:
            result = retry_sync(fn, policy=self._retry_policy, is_retryable=is_retryable_exception)
        except Exception as exc:
            self._breaker.record_failure(exc)
            if is_retryable_exception(exc):
                raise IndexUnavailable("spans_v2 index unavailable") from exc
            raise
        self._breaker.record_success()
        return result

    def _ensure_collection(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return

            def _ensure() -> None:
                from qdrant_client.http import models

                if not self._client.collection_exists(self._collection):
                    distance = getattr(models.Distance, self._config.qdrant.distance.upper(), None)
                    if distance is None:
                        distance = models.Distance.COSINE
                    late_dim = int(self._config.qdrant.late_vector_size)
                    vector_params = {
                        "dense": models.VectorParams(size=self._dim, distance=distance),
                        "late": models.VectorParams(
                            size=late_dim,
                            distance=distance,
                            multivector_config=_multivector_config(models),
                        ),
                    }
                    sparse_config = {"sparse": models.SparseVectorParams()}
                    self._client.create_collection(
                        self._collection,
                        vectors_config=vector_params,
                        sparse_vectors_config=sparse_config,
                    )
                    return
                info = self._client.get_collection(self._collection)
                vectors = info.config.params.vectors
                if isinstance(vectors, dict) and "dense" in vectors:
                    dense_size = getattr(vectors["dense"], "size", None)
                    if dense_size is not None and int(dense_size) != int(self._dim):
                        raise RuntimeError(
                            "spans_v2 dense dim mismatch; change qdrant.spans_v2_collection."
                        )

            self._run(_ensure)
            self._ready = True

    def upsert(self, upserts: list[SpanV2Upsert]) -> None:
        if not upserts:
            return
        self._ensure_collection()
        from qdrant_client.http import models

        points = []
        for item in upserts:
            point_id = f"{item.embedding_model}:{item.capture_id}:{item.span_key}"
            payload = dict(item.payload or {})
            payload.update(
                {
                    "capture_id": item.capture_id,
                    "span_key": item.span_key,
                    "embedding_model": item.embedding_model,
                }
            )
            vector_map: dict[str, object] = {"dense": item.dense_vector}
            if item.late_vectors:
                vector_map["late"] = item.late_vectors
            point_kwargs = {
                "id": point_id,
                "vector": vector_map,
                "payload": payload,
            }
            sparse_vec = item.sparse_vector
            if sparse_vec is not None:
                sparse = models.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values)
                if _supports_sparse_vectors(models):
                    point_kwargs["sparse_vectors"] = {"sparse": sparse}
                else:
                    vector_map["sparse"] = sparse
            points.append(models.PointStruct(**point_kwargs))
        if not points:
            return

        def _upsert() -> None:
            self._client.upsert(self._collection, points=points)

        self._run(_upsert)

    def search_dense(
        self, vector: list[float], k: int, *, filters: dict | None, embedding_model: str
    ) -> list[VectorHit]:
        self._ensure_collection()
        from qdrant_client.http import models

        query_vector: object = ("dense", vector)
        if hasattr(models, "NamedVector"):
            query_vector = models.NamedVector(name="dense", vector=vector)
        filter_obj = _build_filter(models, filters, embedding_model=embedding_model)

        def _search():
            return self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
                query_filter=filter_obj,
            )

        return _hits_from_results(self._run(_search))

    def search_sparse(
        self, vector: SparseEmbedding, k: int, *, filters: dict | None
    ) -> list[VectorHit]:
        self._ensure_collection()
        from qdrant_client.http import models

        sparse = models.SparseVector(indices=vector.indices, values=vector.values)
        if hasattr(models, "NamedSparseVector"):
            query_vector = models.NamedSparseVector(name="sparse", vector=sparse)
        else:
            query_vector = ("sparse", sparse)
        filter_obj = _build_filter(models, filters, embedding_model=None)

        def _search():
            return self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
                query_filter=filter_obj,
            )

        return _hits_from_results(self._run(_search))

    def search_late(
        self, vectors: list[list[float]], k: int, *, filters: dict | None
    ) -> list[VectorHit]:
        self._ensure_collection()
        from qdrant_client.http import models

        query_vector: object = ("late", vectors)
        if hasattr(models, "NamedVector"):
            query_vector = models.NamedVector(name="late", vector=vectors)
        filter_obj = _build_filter(models, filters, embedding_model=None)

        def _search():
            return self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
                query_filter=filter_obj,
            )

        return _hits_from_results(self._run(_search))


def _build_filter(models, filters: dict | None, *, embedding_model: str | None):
    must = []
    if embedding_model:
        must.append(
            models.FieldCondition(
                key="embedding_model", match=models.MatchValue(value=embedding_model)
            )
        )
    if filters:
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, list):
                if value:
                    must.append(models.FieldCondition(key=key, match=models.MatchAny(any=value)))
            else:
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=must)


def _hits_from_results(results) -> list[VectorHit]:
    hits: list[VectorHit] = []
    for hit in results or []:
        payload = hit.payload or {}
        capture_id = str(payload.get("capture_id") or payload.get("event_id") or "")
        span_key = str(payload.get("span_key") or payload.get("span_id") or "")
        if not capture_id or not span_key:
            continue
        hits.append(
            VectorHit(event_id=capture_id, span_key=span_key, score=float(hit.score or 0.0))
        )
    return hits


def _supports_sparse_vectors(models) -> bool:
    fields = getattr(models.PointStruct, "model_fields", None)
    if isinstance(fields, dict):
        return "sparse_vectors" in fields
    return "sparse_vectors" in getattr(models.PointStruct, "__fields__", {})


def _multivector_config(models):
    config_cls = getattr(models, "MultiVectorConfig", None)
    if not config_cls:
        return None
    comparator = getattr(models, "MultiVectorComparator", None)
    if comparator and hasattr(comparator, "MAX_SIM"):
        return config_cls(comparator=comparator.MAX_SIM)
    return config_cls(comparator="max_sim")
