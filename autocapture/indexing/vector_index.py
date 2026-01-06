"""Vector index utilities (Qdrant or local HNSW)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..config import AppConfig
from ..fs_utils import file_lock, safe_replace
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import HNSWMappingRecord


@dataclass(frozen=True)
class VectorHit:
    event_id: str
    span_key: str
    score: float


class QdrantIndex:
    def __init__(self, config: AppConfig, dim: int) -> None:
        self._config = config
        self._client = QdrantClient(url=config.qdrant.url, timeout=2.0)
        self._collection = config.qdrant.collection_name
        self._dim = dim
        self._log = get_logger("index.qdrant")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        distance = getattr(Distance, self._config.qdrant.distance.upper(), Distance.COSINE)
        self._client.create_collection(
            self._collection,
            vectors_config=VectorParams(size=self._dim, distance=distance),
        )

    def upsert_spans(self, items: Iterable[tuple[str, str, list[float], dict]]) -> None:
        points = []
        for event_id, span_key, vector, payload in items:
            point_id = f"{event_id}:{span_key}"
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        if points:
            self._client.upsert(self._collection, points=points)

    def search(self, vector: list[float], limit: int = 20) -> list[VectorHit]:
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        results: list[VectorHit] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                VectorHit(
                    event_id=str(payload.get("event_id", "")),
                    span_key=str(payload.get("span_key", "")),
                    score=float(hit.score or 0.0),
                )
            )
        return results


class HNSWIndex:
    def __init__(self, config: AppConfig, db: DatabaseManager, dim: int) -> None:
        self._config = config
        self._db = db
        self._dim = dim
        self._log = get_logger("index.hnsw")
        self._index_path = Path(config.worker.data_dir) / "embeddings" / "hnsw.index"
        self._lock_path = self._index_path.with_suffix(".lock")
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = None
        self._index = None
        self._load()

    def _load(self) -> None:
        import hnswlib  # type: ignore
        import threading

        self._lock = threading.Lock()
        index = hnswlib.Index(space="cosine", dim=self._dim)
        if self._index_path.exists():
            index.load_index(str(self._index_path))
        else:
            index.init_index(
                max_elements=200000, ef_construction=200, M=16, allow_replace_deleted=True
            )
        index.set_ef(50)
        self._index = index

    def _ensure_capacity(self, count: int) -> None:
        if self._index is None:
            return
        current = self._index.get_current_count()
        max_elements = self._index.get_max_elements()
        if current + count >= max_elements:
            self._index.resize_index(max_elements + max(10000, count * 2))

    def upsert_spans(self, items: Iterable[tuple[str, str, list[float], dict]]) -> None:
        if self._index is None or self._lock is None:
            return
        items_list = list(items)
        if not items_list:
            return
        mapping_records = []
        vectors = []
        with self._db.session() as session:
            for event_id, span_key, vector, payload in items_list:
                mapping = (
                    session.execute(
                        select(HNSWMappingRecord)
                        .where(HNSWMappingRecord.event_id == event_id)
                        .where(HNSWMappingRecord.span_key == span_key)
                    )
                    .scalars()
                    .first()
                )
                if mapping is None:
                    mapping = HNSWMappingRecord(
                        event_id=event_id,
                        span_key=span_key,
                        span_id=payload.get("span_id") if isinstance(payload, dict) else None,
                    )
                    session.add(mapping)
                    try:
                        session.flush()
                    except IntegrityError:
                        session.rollback()
                        mapping = (
                            session.execute(
                                select(HNSWMappingRecord)
                                .where(HNSWMappingRecord.event_id == event_id)
                                .where(HNSWMappingRecord.span_key == span_key)
                            )
                            .scalars()
                            .first()
                        )
                        if mapping is None:
                            raise
                mapping_records.append(mapping)
                vectors.append(vector)

        labels = [record.label for record in mapping_records if record is not None]
        if not labels:
            return
        with self._lock:
            existing_labels = set(self._index.get_ids_list())
            self._ensure_capacity(len(labels))
            for label in labels:
                if label in existing_labels:
                    self._index.mark_deleted(label)
            self._index.add_items(vectors, labels, replace_deleted=True)
            self._save_index()

    def _save_index(self) -> None:
        if self._index is None:
            return
        temp_path = self._index_path.with_suffix(".tmp")
        with file_lock(self._lock_path):
            self._index.save_index(str(temp_path))
            safe_replace(temp_path, self._index_path)

    def search(self, vector: list[float], limit: int = 20) -> list[VectorHit]:
        if self._index is None or self._lock is None:
            return []
        current = self._index.get_current_count()
        if current == 0:
            return []
        k = min(limit, current)
        with self._lock:
            labels, distances = self._index.knn_query([vector], k=k)
        label_list = labels[0].tolist() if hasattr(labels[0], "tolist") else list(labels[0])
        dist_list = distances[0].tolist() if hasattr(distances[0], "tolist") else list(distances[0])
        if not label_list:
            return []
        with self._db.session() as session:
            mappings = (
                session.execute(select(HNSWMappingRecord).where(HNSWMappingRecord.label.in_(label_list)))
                .scalars()
                .all()
            )
        mapping_by_label = {m.label: m for m in mappings}
        hits: list[VectorHit] = []
        for label, distance in zip(label_list, dist_list):
            mapping = mapping_by_label.get(label)
            if not mapping:
                continue
            score = 1.0 - float(distance)
            hits.append(VectorHit(event_id=mapping.event_id, span_key=mapping.span_key, score=score))
        return hits


class VectorIndex:
    def __init__(self, config: AppConfig, db: DatabaseManager, dim: int) -> None:
        self._config = config
        self._db = db
        self._dim = dim
        self._log = get_logger("index.vector")
        self._backend: Optional[object] = None
        self._select_backend()

    def _select_backend(self) -> None:
        try:
            qdrant = QdrantIndex(self._config, self._dim)
            _ = qdrant.search([0.0] * self._dim, limit=1)
            self._backend = qdrant
            self._log.info("Vector index: Qdrant")
            return
        except Exception as exc:
            self._log.warning("Qdrant unavailable; falling back to HNSW: {}", exc)
        self._backend = HNSWIndex(self._config, self._db, self._dim)
        self._log.info("Vector index: HNSW")

    def upsert_spans(self, items: Iterable[tuple[str, str, list[float], dict]]) -> None:
        if self._backend is None:
            return
        self._backend.upsert_spans(items)

    def search(self, vector: list[float], limit: int = 20) -> list[VectorHit]:
        if self._backend is None:
            return []
        return self._backend.search(vector, limit=limit)
