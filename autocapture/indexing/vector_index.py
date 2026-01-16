"""Vector index utilities (Qdrant backend only)."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Optional, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ..config import AppConfig
from ..logging_utils import get_logger
from ..observability.metrics import vector_search_failures_total
from ..resilience import (
    CircuitBreaker,
    RetryPolicy,
    is_retryable_exception,
    retry_sync,
)


class IndexUnavailable(RuntimeError):
    pass


class IndexMisconfigured(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorHit:
    event_id: str
    span_key: str
    score: float


@dataclass(frozen=True)
class SpanEmbeddingUpsert:
    capture_id: str
    span_key: str
    vector: list[float]
    payload: dict
    embedding_model: str


class VectorBackend(Protocol):
    def upsert_spans(self, upserts: list[SpanEmbeddingUpsert]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        k: int,
        *,
        filters: dict | None = None,
        embedding_model: str,
    ) -> list[VectorHit]: ...


class QdrantBackend:
    def __init__(
        self,
        config: AppConfig,
        dim: int,
        *,
        retry_policy: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._config = config
        self._client = QdrantClient(url=config.qdrant.url, timeout=2.0)
        self._collection = config.qdrant.text_collection
        self._dim = dim
        self._log = get_logger("index.qdrant")
        self._retry_policy = retry_policy or RetryPolicy()
        self._breaker = breaker or CircuitBreaker()
        self._collection_ready = False
        self._collection_lock = threading.Lock()

    def allow(self) -> bool:
        return self._breaker.allow()

    def _run_resilient(self, fn):
        if not self._breaker.allow():
            raise IndexUnavailable("vector index unavailable (circuit open)")
        try:
            result = retry_sync(
                fn,
                policy=self._retry_policy,
                is_retryable=is_retryable_exception,
            )
        except Exception as exc:
            self._breaker.record_failure(exc)
            if is_retryable_exception(exc):
                raise IndexUnavailable("vector index unavailable") from exc
            raise
        self._breaker.record_success()
        return result

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        with self._collection_lock:
            if self._collection_ready:
                return

            def _ensure() -> None:
                if not self._client.collection_exists(self._collection):
                    distance = getattr(
                        Distance, self._config.qdrant.distance.upper(), Distance.COSINE
                    )
                    self._client.create_collection(
                        self._collection,
                        vectors_config=VectorParams(size=self._dim, distance=distance),
                    )
                    return
                info = self._client.get_collection(self._collection)
                vectors = info.config.params.vectors
                size = None
                if isinstance(vectors, VectorParams):
                    size = vectors.size
                elif isinstance(vectors, dict) and vectors:
                    first = next(iter(vectors.values()))
                    size = getattr(first, "size", None)
                if size is not None and int(size) != int(self._dim):
                    raise IndexMisconfigured(
                        "Collection dim mismatch; change collection_name or delete collection."
                    )

            self._run_resilient(_ensure)
            self._collection_ready = True

    def upsert_spans(self, upserts: list[SpanEmbeddingUpsert]) -> None:
        if not upserts:
            return
        if not self._breaker.allow():
            raise IndexUnavailable("vector index unavailable (circuit open)")
        self._ensure_collection()
        points: list[PointStruct] = []
        for item in upserts:
            point_id = f"{item.embedding_model}:{item.capture_id}:{item.span_key}"
            payload = dict(item.payload)
            payload.update(
                {
                    "capture_id": item.capture_id,
                    "span_key": item.span_key,
                    "embedding_model": item.embedding_model,
                }
            )
            points.append(PointStruct(id=point_id, vector=item.vector, payload=payload))
        if not points:
            return

        def _upsert() -> None:
            self._client.upsert(self._collection, points=points)

        self._run_resilient(_upsert)

    def search(
        self,
        query_vector: list[float],
        k: int,
        *,
        filters: dict | None = None,
        embedding_model: str,
    ) -> list[VectorHit]:
        if not self._breaker.allow():
            raise IndexUnavailable("vector index unavailable (circuit open)")
        self._ensure_collection()
        must_conditions = [
            FieldCondition(
                key="embedding_model",
                match=MatchValue(value=embedding_model),
            )
        ]
        if filters:
            for key, value in filters.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    if value:
                        must_conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
                else:
                    must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        filter_obj = Filter(must=must_conditions)

        def _search():
            return self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
                query_filter=filter_obj,
            )

        hits = self._run_resilient(_search)
        results: list[VectorHit] = []
        for hit in hits:
            payload = hit.payload or {}
            capture_id = str(payload.get("capture_id") or payload.get("event_id") or "")
            span_key = str(payload.get("span_key") or "")
            if not capture_id or not span_key:
                continue
            results.append(
                VectorHit(
                    event_id=capture_id,
                    span_key=span_key,
                    score=float(hit.score or 0.0),
                )
            )
        return results


class VectorIndex:
    def __init__(
        self,
        config: AppConfig,
        dim: int,
        *,
        backend: Optional[VectorBackend] = None,
    ) -> None:
        self._config = config
        self._dim = dim
        self._log = get_logger("index.vector")
        self._backend: Optional[VectorBackend] = backend
        if self._backend is None:
            if config.qdrant.enabled:
                self._backend = QdrantBackend(config, dim)
                self._log.info("Vector index: Qdrant")
            else:
                self._backend = None
                self._log.info("Vector index: disabled")

    def _backend_allows(self) -> bool:
        backend = self._backend
        if backend is None:
            return False
        allow = getattr(backend, "allow", None)
        return bool(allow() if callable(allow) else True)

    def upsert_spans(self, upserts: list[SpanEmbeddingUpsert]) -> None:
        if self._backend is None:
            return
        if not self._backend_allows():
            raise IndexUnavailable("vector index unavailable (circuit open)")
        self._backend.upsert_spans(upserts)

    def search(
        self,
        query_vector: list[float],
        k: int,
        *,
        filters: dict | None = None,
        embedding_model: str | None = None,
    ) -> list[VectorHit]:
        if self._backend is None:
            return []
        if not self._backend_allows():
            vector_search_failures_total.inc()
            return []
        model = embedding_model or self._config.embed.text_model
        try:
            return self._backend.search(
                query_vector,
                k,
                filters=filters,
                embedding_model=model,
            )
        except IndexUnavailable:
            self._log.warning("Vector index unavailable; returning empty results")
            vector_search_failures_total.inc()
            return []
