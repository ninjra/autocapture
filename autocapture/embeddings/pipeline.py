"""Nightly embedding batch pipeline."""

from __future__ import annotations

import asyncio
import datetime as dt
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer

from ..config import EmbeddingConfig
from ..logging_utils import get_logger
from ..storage import DatabaseManager, EmbeddingRecord, OCRSpanRecord


class EmbeddingBatcher:
    """Generate embeddings for OCR spans and push to Qdrant."""

    def __init__(
        self,
        config: EmbeddingConfig,
        db: DatabaseManager,
        qdrant_client: QdrantClient,
    ) -> None:
        self._config = config
        self._db = db
        self._qdrant = qdrant_client
        self._log = get_logger("embedding")
        self._model: SentenceTransformer | None = None

    async def run_once(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._process_pending)

    def _process_pending(self) -> None:
        model = self._load_model()
        with self._db.session() as session:
            spans = (
                session.query(OCRSpanRecord)
                .outerjoin(EmbeddingRecord, EmbeddingRecord.span_id == OCRSpanRecord.id)
                .filter(EmbeddingRecord.id.is_(None))
                .limit(10000)
                .all()
            )
            if not spans:
                self._log.debug("No spans pending embeddings")
                return

            vectors = model.encode(
                [span.text for span in spans],
                batch_size=self._config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            stored_vectors = (
                vectors.astype(np.float16)
                if self._config.use_half_precision
                else vectors
            )
            qdrant_vectors = vectors.astype(np.float32)

            points = []
            for span, vector in zip(spans, qdrant_vectors, strict=True):
                point_id = f"{span.capture_id}:{span.id}"
                payload = {
                    "capture_id": span.capture_id,
                    "span_id": span.id,
                    "text": span.text,
                    "confidence": span.confidence,
                    "created_at": dt.datetime.utcnow().isoformat(),
                }
                points.append(
                    rest_models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload=payload,
                    )
                )
            self._qdrant.upsert(
                collection_name=self._config.collection_name,
                wait=True,
                points=points,
            )
            for span, vector in zip(spans, stored_vectors, strict=True):
                if self._config.use_half_precision:
                    vector_list = vector.astype(np.float32).tolist()
                else:
                    vector_list = vector.tolist()
                embedding = EmbeddingRecord(
                    capture_id=span.capture_id,
                    span_id=span.id,
                    vector=vector_list,
                    model=self._config.model,
                )
                session.add(embedding)
            self._log.info("Indexed %s spans into Qdrant", len(points))

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._config.model, device="cuda")
            self._log.info("Loaded embedding model %s", self._config.model)
        return self._model
