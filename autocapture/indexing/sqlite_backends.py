"""SQLite-backed vector and spans_v2 backends."""

from __future__ import annotations

import json

from sqlalchemy import bindparam, text

from ..config import AppConfig
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from .spans_v2 import SparseEmbedding, SpanV2Upsert
from .vector_index import SpanEmbeddingUpsert, VectorHit
from .sqlite_utils import (
    cosine_similarity,
    maxsim_score,
    mean_vector,
    signature_bucket,
    vector_from_blob,
    vector_norm,
    vector_signature,
    vector_to_blob,
)


class SqliteVectorBackend:
    def __init__(self, db: DatabaseManager, dim: int, config: AppConfig | None = None) -> None:
        self._db = db
        self._dim = int(dim)
        self._config = config or AppConfig()
        self._log = get_logger("index.sqlite.vector")
        self._signature_seed = 260118
        self._signature_bits = 63
        self._bucket_bits = 16
        self._candidate_factor = 20
        self._candidate_min = 200
        self._candidate_max = 5000
        self._schema_ready = False
        self._last_candidate_strategy: str | None = None
        self._last_candidate_count: int = 0
        self._last_candidate_cap: int = 0
        self._ensure_schema()

    def allow(self) -> bool:
        return True

    @property
    def last_candidate_strategy(self) -> str | None:
        return self._last_candidate_strategy

    @property
    def last_candidate_count(self) -> int:
        return self._last_candidate_count

    @property
    def last_candidate_cap(self) -> int:
        return self._last_candidate_cap

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        engine = self._db.engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS vec_spans ("
                    "point_id TEXT PRIMARY KEY,"
                    "capture_id TEXT NOT NULL,"
                    "span_key TEXT NOT NULL,"
                    "embedding_model TEXT NOT NULL,"
                    "vector BLOB NOT NULL,"
                    "norm REAL NOT NULL,"
                    "signature INTEGER NOT NULL,"
                    "bucket INTEGER NOT NULL,"
                    "app_name TEXT,"
                    "domain TEXT,"
                    "payload_json TEXT"
                    ")"
                )
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_vec_spans_capture ON vec_spans(capture_id)")
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_vec_spans_model ON vec_spans(embedding_model)")
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_vec_spans_bucket ON vec_spans(bucket)")
            )
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_vec_spans_app ON vec_spans(app_name)"))
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_vec_spans_domain ON vec_spans(domain)")
            )
        self._schema_ready = True

    def _candidate_cap(self, k: int) -> int:
        cap = max(k * self._candidate_factor, self._candidate_min)
        return min(cap, self._candidate_max)

    def upsert_spans(self, upserts: list[SpanEmbeddingUpsert]) -> None:
        if not upserts:
            return
        self._ensure_schema()
        engine = self._db.engine
        rows = []
        for item in upserts:
            point_id = f"{item.embedding_model}:{item.capture_id}:{item.span_key}"
            vector = list(item.vector or [])
            blob = vector_to_blob(vector)
            norm = vector_norm(vector)
            signature = vector_signature(
                vector, seed=self._signature_seed, bits=self._signature_bits
            )
            bucket = signature_bucket(signature, bits=self._bucket_bits)
            payload = dict(item.payload or {})
            payload.update(
                {
                    "capture_id": item.capture_id,
                    "span_key": item.span_key,
                    "embedding_model": item.embedding_model,
                }
            )
            rows.append(
                {
                    "point_id": point_id,
                    "capture_id": item.capture_id,
                    "span_key": item.span_key,
                    "embedding_model": item.embedding_model,
                    "vector": blob,
                    "norm": norm,
                    "signature": signature,
                    "bucket": bucket,
                    "app_name": payload.get("app_name"),
                    "domain": payload.get("domain"),
                    "payload_json": json.dumps(payload, separators=(",", ":"), sort_keys=True),
                }
            )
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO vec_spans("
                    "point_id, capture_id, span_key, embedding_model, vector, norm, signature, "
                    "bucket, app_name, domain, payload_json"
                    ") VALUES ("
                    ":point_id, :capture_id, :span_key, :embedding_model, :vector, :norm, "
                    ":signature, :bucket, :app_name, :domain, :payload_json"
                    ") ON CONFLICT(point_id) DO UPDATE SET "
                    "capture_id=excluded.capture_id, "
                    "span_key=excluded.span_key, "
                    "embedding_model=excluded.embedding_model, "
                    "vector=excluded.vector, "
                    "norm=excluded.norm, "
                    "signature=excluded.signature, "
                    "bucket=excluded.bucket, "
                    "app_name=excluded.app_name, "
                    "domain=excluded.domain, "
                    "payload_json=excluded.payload_json"
                ),
                rows,
            )

    def _fetch_candidates(
        self,
        *,
        embedding_model: str,
        signature: int,
        filters: dict | None,
        cap: int,
        use_signature: bool,
    ) -> list[dict]:
        where = ["embedding_model = :model"]
        params: dict[str, object] = {"model": embedding_model, "limit": cap}
        if use_signature:
            bucket = signature_bucket(signature, bits=self._bucket_bits)
            where.append("bucket = :bucket")
            params["bucket"] = bucket
        if filters:
            for key, value in filters.items():
                if value is None:
                    continue
                if key in {"app", "app_name"}:
                    col = "app_name"
                elif key == "domain":
                    col = "domain"
                else:
                    continue
                if isinstance(value, list):
                    if not value:
                        continue
                    where.append(f"{col} IN :{col}")
                    params[col] = list(value)
                else:
                    where.append(f"{col} = :{col}")
                    params[col] = value
        clause = " AND ".join(where)
        stmt = text(
            "SELECT capture_id, span_key, vector, norm FROM vec_spans "
            f"WHERE {clause} ORDER BY capture_id, span_key LIMIT :limit"
        )
        for key, value in list(params.items()):
            if isinstance(value, list):
                stmt = stmt.bindparams(bindparam(key, expanding=True))
        with self._db.engine.begin() as conn:
            rows = conn.execute(stmt, params).fetchall()
        return [
            {
                "capture_id": row[0],
                "span_key": row[1],
                "vector": row[2],
                "norm": float(row[3] or 0.0),
            }
            for row in rows
        ]

    def search(
        self,
        query_vector: list[float],
        k: int,
        *,
        filters: dict | None = None,
        embedding_model: str,
    ) -> list[VectorHit]:
        self._ensure_schema()
        vector = list(query_vector or [])
        signature = vector_signature(vector, seed=self._signature_seed, bits=self._signature_bits)
        cap = self._candidate_cap(k)
        self._last_candidate_cap = cap
        candidates = self._fetch_candidates(
            embedding_model=embedding_model,
            signature=signature,
            filters=filters,
            cap=cap,
            use_signature=True,
        )
        self._last_candidate_strategy = "signature"
        if len(candidates) < max(k, 1):
            fallback = self._fetch_candidates(
                embedding_model=embedding_model,
                signature=signature,
                filters=filters,
                cap=cap,
                use_signature=False,
            )
            if fallback:
                candidates = fallback
                self._last_candidate_strategy = "fallback"
        self._last_candidate_count = len(candidates)
        if not candidates:
            return []
        query_norm = vector_norm(vector)
        hits: list[VectorHit] = []
        for item in candidates:
            stored_vector = vector_from_blob(item["vector"])
            score = cosine_similarity(
                vector, stored_vector, left_norm=query_norm, right_norm=item["norm"]
            )
            hits.append(
                VectorHit(
                    event_id=str(item["capture_id"]),
                    span_key=str(item["span_key"]),
                    score=score,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.event_id, hit.span_key))
        return hits[:k]

    def delete_event_ids(self, event_ids: list[str]) -> int:
        if not event_ids:
            return 0
        self._ensure_schema()
        stmt = text("DELETE FROM vec_spans WHERE capture_id IN :event_ids").bindparams(
            bindparam("event_ids", expanding=True)
        )
        with self._db.engine.begin() as conn:
            result = conn.execute(stmt, {"event_ids": list(event_ids)})
        return int(result.rowcount or 0)

    def list_event_ids(self) -> list[str]:
        self._ensure_schema()
        with self._db.engine.begin() as conn:
            rows = conn.execute(text("SELECT DISTINCT capture_id FROM vec_spans")).fetchall()
        return [str(row[0]) for row in rows if row and row[0]]


class SqliteSpansV2Backend:
    def __init__(self, db: DatabaseManager, dim: int, config: AppConfig | None = None) -> None:
        self._db = db
        self._dim = int(dim)
        self._config = config or AppConfig()
        self._log = get_logger("index.sqlite.spans_v2")
        self._signature_seed = 260118
        self._signature_bits = 63
        self._bucket_bits = 16
        self._schema_ready = False
        self._last_candidate_strategy: str | None = None
        self._last_candidate_count: int = 0
        self._last_candidate_cap: int = 0
        self._ensure_schema()

    def allow(self) -> bool:
        return True

    @property
    def last_candidate_strategy(self) -> str | None:
        return self._last_candidate_strategy

    @property
    def last_candidate_count(self) -> int:
        return self._last_candidate_count

    @property
    def last_candidate_cap(self) -> int:
        return self._last_candidate_cap

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        engine = self._db.engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS vec_spans_v2_meta ("
                    "span_id TEXT PRIMARY KEY,"
                    "capture_id TEXT NOT NULL,"
                    "span_key TEXT NOT NULL,"
                    "embedding_model TEXT NOT NULL,"
                    "app TEXT,"
                    "domain TEXT,"
                    "window_title TEXT,"
                    "frame_id TEXT,"
                    "frame_hash TEXT,"
                    "ts TEXT,"
                    "bbox_norm_json TEXT,"
                    "text TEXT,"
                    "tags_json TEXT"
                    ")"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS vec_spans_v2_dense ("
                    "span_id TEXT PRIMARY KEY,"
                    "vector BLOB NOT NULL,"
                    "norm REAL NOT NULL,"
                    "signature INTEGER NOT NULL,"
                    "bucket INTEGER NOT NULL,"
                    "FOREIGN KEY(span_id) REFERENCES vec_spans_v2_meta(span_id) ON DELETE CASCADE"
                    ")"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS vec_spans_v2_sparse ("
                    "span_id TEXT NOT NULL,"
                    "token_id INTEGER NOT NULL,"
                    "weight REAL NOT NULL,"
                    "PRIMARY KEY (span_id, token_id),"
                    "FOREIGN KEY(span_id) REFERENCES vec_spans_v2_meta(span_id) ON DELETE CASCADE"
                    ")"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS vec_spans_v2_late ("
                    "span_id TEXT NOT NULL,"
                    "token_index INTEGER NOT NULL,"
                    "vector BLOB NOT NULL,"
                    "norm REAL NOT NULL,"
                    "PRIMARY KEY (span_id, token_index),"
                    "FOREIGN KEY(span_id) REFERENCES vec_spans_v2_meta(span_id) ON DELETE CASCADE"
                    ")"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_capture ON "
                    "vec_spans_v2_meta(capture_id)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_model ON "
                    "vec_spans_v2_meta(embedding_model)"
                )
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_app ON vec_spans_v2_meta(app)")
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_domain ON vec_spans_v2_meta(domain)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_bucket ON vec_spans_v2_dense(bucket)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_vec_spans_v2_sparse_token ON "
                    "vec_spans_v2_sparse(token_id)"
                )
            )
        self._schema_ready = True

    def _candidate_cap(self, k: int) -> int:
        base = max(k, int(self._config.retrieval.late_candidate_k))
        return max(base, 50)

    def upsert(self, upserts: list[SpanV2Upsert]) -> None:
        if not upserts:
            return
        self._ensure_schema()
        engine = self._db.engine
        with engine.begin() as conn:
            for item in upserts:
                span_id = f"{item.embedding_model}:{item.capture_id}:{item.span_key}"
                payload = dict(item.payload or {})
                meta = {
                    "span_id": span_id,
                    "capture_id": item.capture_id,
                    "span_key": item.span_key,
                    "embedding_model": item.embedding_model,
                    "app": payload.get("app"),
                    "domain": payload.get("domain"),
                    "window_title": payload.get("window_title"),
                    "frame_id": payload.get("frame_id"),
                    "frame_hash": payload.get("frame_hash"),
                    "ts": payload.get("ts"),
                    "bbox_norm_json": json.dumps(payload.get("bbox_norm"), separators=(",", ":")),
                    "text": payload.get("text"),
                    "tags_json": json.dumps(
                        payload.get("tags") or {}, separators=(",", ":"), sort_keys=True
                    ),
                }
                conn.execute(
                    text(
                        "INSERT INTO vec_spans_v2_meta("
                        "span_id, capture_id, span_key, embedding_model, app, domain, "
                        "window_title, frame_id, frame_hash, ts, bbox_norm_json, text, tags_json"
                        ") VALUES ("
                        ":span_id, :capture_id, :span_key, :embedding_model, :app, :domain, "
                        ":window_title, :frame_id, :frame_hash, :ts, :bbox_norm_json, :text, :tags_json"
                        ") ON CONFLICT(span_id) DO UPDATE SET "
                        "capture_id=excluded.capture_id, "
                        "span_key=excluded.span_key, "
                        "embedding_model=excluded.embedding_model, "
                        "app=excluded.app, "
                        "domain=excluded.domain, "
                        "window_title=excluded.window_title, "
                        "frame_id=excluded.frame_id, "
                        "frame_hash=excluded.frame_hash, "
                        "ts=excluded.ts, "
                        "bbox_norm_json=excluded.bbox_norm_json, "
                        "text=excluded.text, "
                        "tags_json=excluded.tags_json"
                    ),
                    meta,
                )
                dense_vector = list(item.dense_vector or [])
                dense_blob = vector_to_blob(dense_vector)
                dense_norm = vector_norm(dense_vector)
                dense_signature = vector_signature(
                    dense_vector, seed=self._signature_seed, bits=self._signature_bits
                )
                dense_bucket = signature_bucket(dense_signature, bits=self._bucket_bits)
                conn.execute(
                    text(
                        "INSERT INTO vec_spans_v2_dense("
                        "span_id, vector, norm, signature, bucket"
                        ") VALUES ("
                        ":span_id, :vector, :norm, :signature, :bucket"
                        ") ON CONFLICT(span_id) DO UPDATE SET "
                        "vector=excluded.vector, "
                        "norm=excluded.norm, "
                        "signature=excluded.signature, "
                        "bucket=excluded.bucket"
                    ),
                    {
                        "span_id": span_id,
                        "vector": dense_blob,
                        "norm": dense_norm,
                        "signature": dense_signature,
                        "bucket": dense_bucket,
                    },
                )
                conn.execute(
                    text("DELETE FROM vec_spans_v2_sparse WHERE span_id = :span_id"),
                    {"span_id": span_id},
                )
                sparse = item.sparse_vector
                if sparse is not None and sparse.indices and sparse.values:
                    rows = [
                        {
                            "span_id": span_id,
                            "token_id": int(token_id),
                            "weight": float(weight),
                        }
                        for token_id, weight in zip(sparse.indices, sparse.values)
                    ]
                    conn.execute(
                        text(
                            "INSERT INTO vec_spans_v2_sparse(span_id, token_id, weight) "
                            "VALUES (:span_id, :token_id, :weight)"
                        ),
                        rows,
                    )
                conn.execute(
                    text("DELETE FROM vec_spans_v2_late WHERE span_id = :span_id"),
                    {"span_id": span_id},
                )
                if item.late_vectors:
                    rows = [
                        {
                            "span_id": span_id,
                            "token_index": idx,
                            "vector": vector_to_blob(list(vec)),
                            "norm": vector_norm(vec),
                        }
                        for idx, vec in enumerate(item.late_vectors)
                    ]
                    conn.execute(
                        text(
                            "INSERT INTO vec_spans_v2_late(span_id, token_index, vector, norm) "
                            "VALUES (:span_id, :token_index, :vector, :norm)"
                        ),
                        rows,
                    )

    def _fetch_dense_candidates(
        self,
        *,
        signature: int,
        embedding_model: str,
        filters: dict | None,
        cap: int,
        use_signature: bool,
    ) -> list[dict]:
        where = ["m.embedding_model = :model"]
        params: dict[str, object] = {"model": embedding_model, "limit": cap}
        if use_signature:
            bucket = signature_bucket(signature, bits=self._bucket_bits)
            where.append("d.bucket = :bucket")
            params["bucket"] = bucket
        if filters:
            for key, value in filters.items():
                if value is None:
                    continue
                if key in {"app", "app_name"}:
                    col = "m.app"
                elif key == "domain":
                    col = "m.domain"
                else:
                    continue
                if isinstance(value, list):
                    if not value:
                        continue
                    where.append(f"{col} IN :{col.replace('.', '_')}")
                    params[col.replace(".", "_")] = list(value)
                else:
                    where.append(f"{col} = :{col.replace('.', '_')}")
                    params[col.replace(".", "_")] = value
        clause = " AND ".join(where)
        stmt = text(
            "SELECT m.capture_id, m.span_key, d.vector, d.norm, d.span_id "
            "FROM vec_spans_v2_dense d "
            "JOIN vec_spans_v2_meta m ON m.span_id = d.span_id "
            f"WHERE {clause} ORDER BY m.capture_id, m.span_key LIMIT :limit"
        )
        for key, value in list(params.items()):
            if isinstance(value, list):
                stmt = stmt.bindparams(bindparam(key, expanding=True))
        with self._db.engine.begin() as conn:
            rows = conn.execute(stmt, params).fetchall()
        return [
            {
                "capture_id": row[0],
                "span_key": row[1],
                "vector": row[2],
                "norm": float(row[3] or 0.0),
                "span_id": row[4],
            }
            for row in rows
        ]

    def search_dense(
        self, vector: list[float], k: int, *, filters: dict | None, embedding_model: str
    ) -> list[VectorHit]:
        self._ensure_schema()
        signature = vector_signature(vector, seed=self._signature_seed, bits=self._signature_bits)
        cap = self._candidate_cap(k)
        self._last_candidate_cap = cap
        candidates = self._fetch_dense_candidates(
            signature=signature,
            embedding_model=embedding_model,
            filters=filters,
            cap=cap,
            use_signature=True,
        )
        self._last_candidate_strategy = "signature"
        if len(candidates) < max(k, 1):
            fallback = self._fetch_dense_candidates(
                signature=signature,
                embedding_model=embedding_model,
                filters=filters,
                cap=cap,
                use_signature=False,
            )
            if fallback:
                candidates = fallback
                self._last_candidate_strategy = "fallback"
        self._last_candidate_count = len(candidates)
        if not candidates:
            return []
        query_norm = vector_norm(vector)
        hits: list[VectorHit] = []
        for item in candidates:
            stored_vector = vector_from_blob(item["vector"])
            score = cosine_similarity(
                vector, stored_vector, left_norm=query_norm, right_norm=item["norm"]
            )
            hits.append(
                VectorHit(
                    event_id=str(item["capture_id"]),
                    span_key=str(item["span_key"]),
                    score=score,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.event_id, hit.span_key))
        return hits[:k]

    def search_sparse(
        self, vector: SparseEmbedding, k: int, *, filters: dict | None
    ) -> list[VectorHit]:
        self._ensure_schema()
        if not vector.indices or not vector.values:
            return []
        pairs = list(zip(vector.indices, vector.values))
        values_clause = ", ".join(f"(:token_{idx}, :weight_{idx})" for idx in range(len(pairs)))
        params: dict[str, object] = {
            f"token_{idx}": int(token_id) for idx, (token_id, _weight) in enumerate(pairs)
        }
        params.update(
            {f"weight_{idx}": float(weight) for idx, (_token_id, weight) in enumerate(pairs)}
        )
        where = []
        if filters:
            for key, value in filters.items():
                if value is None:
                    continue
                if key in {"app", "app_name"}:
                    col = "m.app"
                elif key == "domain":
                    col = "m.domain"
                else:
                    continue
                if isinstance(value, list):
                    if not value:
                        continue
                    where.append(f"{col} IN :{col.replace('.', '_')}")
                    params[col.replace(".", "_")] = list(value)
                else:
                    where.append(f"{col} = :{col.replace('.', '_')}")
                    params[col.replace(".", "_")] = value
        where_clause = ""
        if where:
            where_clause = " AND " + " AND ".join(where)
        sql = (
            "WITH query(token_id, weight) AS (VALUES "
            f"{values_clause}"
            ") "
            "SELECT m.capture_id, m.span_key, s.span_id, SUM(s.weight * query.weight) AS score "
            "FROM vec_spans_v2_sparse s "
            "JOIN query ON s.token_id = query.token_id "
            "JOIN vec_spans_v2_meta m ON m.span_id = s.span_id "
            f"WHERE 1=1{where_clause} "
            "GROUP BY s.span_id "
            "ORDER BY score DESC, m.capture_id, m.span_key "
            "LIMIT :limit"
        )
        params["limit"] = int(k)
        stmt = text(sql)
        for key, value in list(params.items()):
            if isinstance(value, list):
                stmt = stmt.bindparams(bindparam(key, expanding=True))
        with self._db.engine.begin() as conn:
            rows = conn.execute(stmt, params).fetchall()
        hits = [
            VectorHit(event_id=str(row[0]), span_key=str(row[1]), score=float(row[3] or 0.0))
            for row in rows
        ]
        return hits

    def search_late(
        self, vectors: list[list[float]], k: int, *, filters: dict | None
    ) -> list[VectorHit]:
        self._ensure_schema()
        if not vectors:
            return []
        candidate_cap = self._candidate_cap(k)
        self._last_candidate_cap = candidate_cap
        query_mean = mean_vector(vectors)
        signature = vector_signature(
            query_mean, seed=self._signature_seed, bits=self._signature_bits
        )
        candidates = self._fetch_dense_candidates(
            signature=signature,
            embedding_model=self._config.embed.text_model,
            filters=filters,
            cap=candidate_cap,
            use_signature=True,
        )
        self._last_candidate_strategy = "signature"
        if len(candidates) < max(k, 1):
            fallback = self._fetch_dense_candidates(
                signature=signature,
                embedding_model=self._config.embed.text_model,
                filters=filters,
                cap=candidate_cap,
                use_signature=False,
            )
            if fallback:
                candidates = fallback
                self._last_candidate_strategy = "fallback"
        self._last_candidate_count = len(candidates)
        if not candidates:
            return []
        span_ids = [item["span_id"] for item in candidates]
        stmt = text(
            "SELECT span_id, vector, norm FROM vec_spans_v2_late "
            "WHERE span_id IN :span_ids ORDER BY span_id, token_index"
        ).bindparams(bindparam("span_ids", expanding=True))
        late_vectors: dict[str, list[list[float]]] = {}
        with self._db.engine.begin() as conn:
            rows = conn.execute(stmt, {"span_ids": span_ids}).fetchall()
        for span_id, blob, _norm in rows:
            late_vectors.setdefault(span_id, []).append(vector_from_blob(blob))
        hits: list[VectorHit] = []
        for item in candidates:
            doc_vectors = late_vectors.get(item["span_id"], [])
            score = maxsim_score(vectors, doc_vectors)
            hits.append(
                VectorHit(
                    event_id=str(item["capture_id"]),
                    span_key=str(item["span_key"]),
                    score=score,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.event_id, hit.span_key))
        return hits[:k]

    def delete_event_ids(self, event_ids: list[str]) -> int:
        if not event_ids:
            return 0
        self._ensure_schema()
        stmt = text("DELETE FROM vec_spans_v2_meta WHERE capture_id IN :event_ids").bindparams(
            bindparam("event_ids", expanding=True)
        )
        with self._db.engine.begin() as conn:
            result = conn.execute(stmt, {"event_ids": list(event_ids)})
        return int(result.rowcount or 0)

    def list_event_ids(self) -> list[str]:
        self._ensure_schema()
        with self._db.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT capture_id FROM vec_spans_v2_meta")
            ).fetchall()
        return [str(row[0]) for row in rows if row and row[0]]
