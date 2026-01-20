"""Postgres-backed Memory Service store."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
import sqlalchemy as sa

from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..config import MemoryServiceConfig
from .policy import MemoryPolicyValidator
from .providers import Embedder, Reranker
from .schemas import (
    MemoryCard,
    MemoryFeedbackRequest,
    MemoryFeedbackResponse,
    MemoryIngestRequest,
    MemoryIngestResponse,
    MemoryProposal,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryRejectDetail,
    ProvenancePointer,
)
from .utils import (
    canonicalize_text,
    estimate_tokens,
    hash_text,
    stable_memory_id,
    utc_now,
    vector_literal,
)

_LOG = get_logger("memory.service")


@dataclass
class CandidateScores:
    semantic: float = 0.0
    keyword: float = 0.0
    graph: float = 0.0
    rerank: float = 0.0
    reasons: set[str] = field(default_factory=set)


@dataclass
class CandidateItem:
    memory_id: str
    memory_type: str
    content_text: str
    content_json: dict
    importance: float
    trust_tier: float
    created_at: dt.datetime
    sensitivity_rank: int
    scores: CandidateScores
    score: float = 0.0


class MemoryServiceStore:
    def __init__(
        self,
        db: DatabaseManager,
        config: MemoryServiceConfig,
        embedder: Embedder,
        reranker: Reranker | None,
    ) -> None:
        self._db = db
        self._config = config
        self._embedder = embedder
        self._reranker = reranker
        self._policy = MemoryPolicyValidator(config.policy)
        self._ensure_rls()

    def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResponse:
        self._require_postgres()
        now = request.debug.now_override if request.debug else None
        now = _ensure_aware(now) if now else utc_now()
        namespace = (request.namespace or self._config.default_namespace or "default").strip()
        accepted = 0
        deduped = 0
        rejects: list[MemoryRejectDetail] = []
        with self._db.session() as session:
            self._set_rls_namespace(session, namespace)
            for idx, proposal in enumerate(request.proposals):
                reasons = self._policy.validate_proposal(proposal)
                if not proposal.provenance:
                    reasons.append("missing_provenance")
                provenance_issue = self._validate_provenance(session, proposal, namespace)
                if provenance_issue:
                    reasons.append(provenance_issue)
                if reasons:
                    rejects.append(MemoryRejectDetail(index=idx, reasons=sorted(set(reasons))))
                    continue

                content_text = canonicalize_text(proposal.content_text)
                if not content_text:
                    rejects.append(MemoryRejectDetail(index=idx, reasons=["content_empty"]))
                    continue
                content_hash = hash_text(content_text)
                memory_id = stable_memory_id(namespace, proposal.memory_type, content_hash)

                exists = session.execute(
                    sa.text("SELECT 1 FROM memory_items WHERE memory_id = :memory_id"),
                    {"memory_id": memory_id},
                ).first()
                if exists:
                    deduped += 1
                else:
                    self._insert_memory_item(session, proposal, namespace, memory_id, content_text, now)
                    accepted += 1

                self._insert_provenance(session, proposal, namespace, memory_id, now)
                self._upsert_entities(session, proposal, namespace, memory_id, now)
                self._insert_embedding(session, proposal, memory_id, content_text, now)

        return MemoryIngestResponse(
            request_id=request.request_id,
            accepted=accepted,
            deduped=deduped,
            rejected=len(rejects),
            rejects=rejects,
        )

    def query(self, request: MemoryQueryRequest) -> MemoryQueryResponse:
        self._require_postgres()
        namespace = (request.namespace or self._config.default_namespace or "default").strip()
        policy_reasons = self._policy.validate_query_policy(request.policy)
        if policy_reasons:
            raise ValueError(f"policy_rejected: {', '.join(sorted(policy_reasons))}")
        now = _ensure_aware(request.now_override) if request.now_override else utc_now()
        candidates = self._gather_candidates(request, namespace, now)
        ranked = self._rank_candidates(candidates, now, request.query, namespace=namespace)
        cards, truncated = self._pack_cards(ranked, request, namespace=namespace)
        return MemoryQueryResponse(
            request_id=request.request_id,
            cards=cards,
            warnings=[],
            truncated=truncated,
        )

    def feedback(self, request: MemoryFeedbackRequest) -> MemoryFeedbackResponse:
        self._require_postgres()
        namespace = (request.namespace or self._config.default_namespace or "default").strip()
        created_at = utc_now()
        feedback_id = stable_memory_id(namespace, "feedback", f"{request.memory_id}:{created_at}")
        with self._db.session() as session:
            self._set_rls_namespace(session, namespace)
            session.execute(
                sa.text(
                    "INSERT INTO memory_feedback("
                    "feedback_id, memory_id, namespace, useful, reason, request_id, created_at"
                    ") VALUES (:feedback_id, :memory_id, :namespace, :useful, :reason, :request_id, :created_at)"
                ),
                {
                    "feedback_id": feedback_id,
                    "memory_id": request.memory_id,
                    "namespace": namespace,
                    "useful": bool(request.useful),
                    "reason": request.reason,
                    "request_id": request.request_id,
                    "created_at": created_at,
                },
            )
        return MemoryFeedbackResponse(request_id=request.request_id, stored=True)

    def health(self) -> dict[str, object]:
        self._require_postgres()
        warnings: list[str] = []
        with self._db.engine.connect() as conn:
            db_connected = conn.execute(sa.text("SELECT 1")).scalar() == 1
            pgvector = (
                conn.execute(
                    sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"),
                ).first()
                is not None
            )
            tables_ready = (
                conn.execute(sa.text("SELECT to_regclass('public.memory_items')"))
                .scalar()
                is not None
            )
        if not pgvector:
            warnings.append("pgvector_missing")
        if not tables_ready:
            warnings.append("tables_missing")
        return {
            "status": "ok" if db_connected else "error",
            "db_connected": db_connected,
            "pgvector": pgvector,
            "tables_ready": tables_ready,
            "warnings": warnings,
        }

    def _require_postgres(self) -> None:
        dialect = self._db.engine.dialect.name
        if dialect != "postgresql":
            raise RuntimeError("Memory Service requires PostgreSQL")

    def _ensure_rls(self) -> None:
        if not self._config.enable_rls:
            return
        try:
            with self._db.engine.begin() as conn:
                if conn.dialect.name != "postgresql":
                    return
                tables = [
                    "artifact_versions",
                    "artifact_chunks",
                    "memory_items",
                    "entities",
                    "edges",
                    "memory_feedback",
                ]
                for table in tables:
                    conn.execute(sa.text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
                    exists = conn.execute(
                        sa.text(
                            "SELECT 1 FROM pg_policies "
                            "WHERE schemaname = 'public' "
                            "AND tablename = :table "
                            "AND policyname = :policy"
                        ),
                        {"table": table, "policy": "memory_namespace_isolation"},
                    ).first()
                    if exists is None:
                        policy_sql = (
                            "CREATE POLICY memory_namespace_isolation ON "
                            f"{table} USING (namespace = "
                            "current_setting('autocapture.namespace', true))"
                        )
                        conn.execute(
                            sa.text(policy_sql)
                        )
        except Exception as exc:  # pragma: no cover - best effort setup
            _LOG.warning("Failed to enable memory service RLS: {}", exc)

    def _set_rls_namespace(self, session, namespace: str) -> None:
        if not self._config.enable_rls:
            return
        try:
            session.execute(
                sa.text("SELECT set_config('autocapture.namespace', :namespace, true)"),
                {"namespace": namespace},
            )
        except Exception as exc:  # pragma: no cover - best effort
            _LOG.warning("Failed to set memory service namespace: {}", exc)

    def _validate_provenance(
        self, session, proposal: MemoryProposal, namespace: str
    ) -> str | None:
        for pointer in proposal.provenance:
            row = session.execute(
                sa.text(
                    "SELECT excerpt_hash FROM artifact_chunks "
                    "WHERE chunk_id = :chunk_id "
                    "AND artifact_version_id = :artifact_version_id "
                    "AND namespace = :namespace"
                ),
                {
                    "chunk_id": pointer.chunk_id,
                    "artifact_version_id": pointer.artifact_version_id,
                    "namespace": namespace,
                },
            ).first()
            if row is None:
                return "orphan_provenance"
            if row[0] != pointer.excerpt_hash:
                return "excerpt_hash_mismatch"
        return None

    def _insert_memory_item(
        self,
        session,
        proposal: MemoryProposal,
        namespace: str,
        memory_id: str,
        content_text: str,
        now: dt.datetime,
    ) -> None:
        sensitivity_rank = self._config.policy.sensitivity_order.index(
            proposal.policy.sensitivity
        )
        audiences = sorted({aud for aud in proposal.policy.audience if aud})
        session.execute(
            sa.text(
                "INSERT INTO memory_items("
                "memory_id, namespace, memory_type, content_text, content_json, content_hash, "
                "status, importance, trust_tier, audiences, sensitivity, sensitivity_rank, "
                "valid_from, valid_to, policy_labels, created_at, updated_at"
                ") VALUES ("
                ":memory_id, :namespace, :memory_type, :content_text, :content_json, :content_hash, "
                ":status, :importance, :trust_tier, :audiences, :sensitivity, :sensitivity_rank, "
                ":valid_from, :valid_to, :policy_labels, :created_at, :updated_at"
                ") ON CONFLICT (memory_id) DO NOTHING"
            ),
            {
                "memory_id": memory_id,
                "namespace": namespace,
                "memory_type": proposal.memory_type,
                "content_text": content_text,
                "content_json": proposal.content_json,
                "content_hash": hash_text(content_text),
                "status": "validated",
                "importance": float(proposal.importance),
                "trust_tier": float(proposal.trust),
                "audiences": audiences,
                "sensitivity": proposal.policy.sensitivity,
                "sensitivity_rank": sensitivity_rank,
                "valid_from": proposal.valid_from,
                "valid_to": proposal.valid_to,
                "policy_labels": {
                    "audience": audiences,
                    "sensitivity": proposal.policy.sensitivity,
                    "labels": proposal.policy.labels,
                },
                "created_at": now,
                "updated_at": now,
            },
        )

    def _insert_provenance(
        self,
        session,
        proposal: MemoryProposal,
        namespace: str,
        memory_id: str,
        now: dt.datetime,
    ) -> None:
        for pointer in proposal.provenance:
            session.execute(
                sa.text(
                    "INSERT INTO memory_provenance("
                    "memory_id, artifact_version_id, chunk_id, start_offset, end_offset, excerpt_hash, created_at"
                    ") VALUES ("
                    ":memory_id, :artifact_version_id, :chunk_id, :start_offset, :end_offset, :excerpt_hash, :created_at"
                    ") ON CONFLICT (memory_id, chunk_id) DO NOTHING"
                ),
                {
                    "memory_id": memory_id,
                    "artifact_version_id": pointer.artifact_version_id,
                    "chunk_id": pointer.chunk_id,
                    "start_offset": pointer.start_offset,
                    "end_offset": pointer.end_offset,
                    "excerpt_hash": pointer.excerpt_hash,
                    "created_at": now,
                },
            )

    def _upsert_entities(
        self,
        session,
        proposal: MemoryProposal,
        namespace: str,
        memory_id: str,
        now: dt.datetime,
    ) -> None:
        for entity in proposal.entities:
            entity_id = stable_memory_id(namespace, "entity", f"{entity.kind}:{entity.name}")
            session.execute(
                sa.text(
                    "INSERT INTO entities(entity_id, namespace, kind, name, created_at) "
                    "VALUES (:entity_id, :namespace, :kind, :name, :created_at) "
                    "ON CONFLICT (namespace, kind, name) DO NOTHING"
                ),
                {
                    "entity_id": entity_id,
                    "namespace": namespace,
                    "kind": entity.kind,
                    "name": entity.name,
                    "created_at": now,
                },
            )
            session.execute(
                sa.text(
                    "INSERT INTO memory_entities(memory_id, entity_id) "
                    "VALUES (:memory_id, :entity_id) ON CONFLICT DO NOTHING"
                ),
                {"memory_id": memory_id, "entity_id": entity_id},
            )

    def _insert_embedding(
        self,
        session,
        proposal: MemoryProposal,
        memory_id: str,
        content_text: str,
        now: dt.datetime,
    ) -> None:
        if not self._config.enable_query_embedding:
            return
        vectors = self._embedder.embed_texts([content_text])
        if not vectors:
            return
        embedding = vector_literal(vectors[0])
        session.execute(
            sa.text(
                "INSERT INTO memory_embeddings(memory_id, model, embedding, created_at) "
                "VALUES (:memory_id, :model, :embedding::vector, :created_at) "
                "ON CONFLICT (memory_id, model) DO NOTHING"
            ),
            {
                "memory_id": memory_id,
                "model": self._embedder.model_id,
                "embedding": embedding,
                "created_at": now,
            },
        )

    def _gather_candidates(
        self, request: MemoryQueryRequest, namespace: str, now: dt.datetime
    ) -> dict[str, CandidateScores]:
        scores: dict[str, CandidateScores] = {}
        max_rank = self._config.policy.sensitivity_order.index(request.policy.sensitivity_max)
        allowed_statuses = ["active", "validated"]

        def _record(memory_id: str, key: str, value: float, reason: str) -> None:
            entry = scores.setdefault(memory_id, CandidateScores())
            setattr(entry, key, max(getattr(entry, key), value))
            entry.reasons.add(reason)

        with self._db.session() as session:
            self._set_rls_namespace(session, namespace)
            if self._config.enable_query_embedding and request.query_embedding:
                embedding_literal = vector_literal(request.query_embedding)
            elif self._config.enable_query_embedding:
                embedding_literal = vector_literal(self._embedder.embed_texts([request.query])[0])
            else:
                embedding_literal = None

            if embedding_literal and self._config.retrieval.topk_vector > 0:
                rows = session.execute(
                    sa.text(
                        "SELECT mi.memory_id, (me.embedding <-> :embedding::vector) AS distance "
                        "FROM memory_items mi "
                        "JOIN memory_embeddings me ON mi.memory_id = me.memory_id "
                        "WHERE mi.namespace = :namespace "
                        "AND mi.status = ANY(:statuses) "
                        "AND mi.sensitivity_rank <= :max_rank "
                        "AND (mi.valid_from IS NULL OR mi.valid_from <= :now) "
                        "AND (mi.valid_to IS NULL OR mi.valid_to >= :now) "
                        "AND mi.audiences && :audiences "
                        "ORDER BY distance ASC, mi.memory_id ASC "
                        "LIMIT :limit"
                    ),
                    {
                        "embedding": embedding_literal,
                        "namespace": namespace,
                        "statuses": allowed_statuses,
                        "max_rank": max_rank,
                        "now": now,
                        "audiences": request.policy.audience,
                        "limit": request.topk_vector or self._config.retrieval.topk_vector,
                    },
                ).fetchall()
                for memory_id, distance in rows:
                    sem_score = 1.0 / (1.0 + float(distance))
                    _record(memory_id, "semantic", sem_score, "semantic")

            if self._config.retrieval.topk_keyword > 0:
                rows = session.execute(
                    sa.text(
                        "SELECT mi.memory_id, ts_rank_cd(mi.content_tsv, "
                        "plainto_tsquery('english', :query)) AS rank "
                        "FROM memory_items mi "
                        "WHERE mi.namespace = :namespace "
                        "AND mi.status = ANY(:statuses) "
                        "AND mi.sensitivity_rank <= :max_rank "
                        "AND (mi.valid_from IS NULL OR mi.valid_from <= :now) "
                        "AND (mi.valid_to IS NULL OR mi.valid_to >= :now) "
                        "AND mi.audiences && :audiences "
                        "AND mi.content_tsv @@ plainto_tsquery('english', :query) "
                        "ORDER BY rank DESC, mi.memory_id ASC "
                        "LIMIT :limit"
                    ),
                    {
                        "query": request.query,
                        "namespace": namespace,
                        "statuses": allowed_statuses,
                        "max_rank": max_rank,
                        "now": now,
                        "audiences": request.policy.audience,
                        "limit": request.topk_keyword or self._config.retrieval.topk_keyword,
                    },
                ).fetchall()
                for memory_id, rank in rows:
                    _record(memory_id, "keyword", float(rank), "keyword")

            if request.entity_hints and self._config.retrieval.topk_graph > 0:
                entity_ids = self._resolve_entity_ids(session, namespace, request.entity_hints)
                graph_ids = self._expand_graph(session, namespace, entity_ids)
                if graph_ids:
                    rows = session.execute(
                        sa.text(
                            "SELECT DISTINCT me.memory_id "
                            "FROM memory_entities me "
                            "JOIN memory_items mi ON mi.memory_id = me.memory_id "
                            "WHERE me.entity_id = ANY(:entity_ids) "
                            "AND mi.namespace = :namespace "
                            "AND mi.status = ANY(:statuses) "
                            "AND mi.sensitivity_rank <= :max_rank "
                            "AND (mi.valid_from IS NULL OR mi.valid_from <= :now) "
                            "AND (mi.valid_to IS NULL OR mi.valid_to >= :now) "
                            "AND mi.audiences && :audiences "
                            "LIMIT :limit"
                        ),
                        {
                            "entity_ids": graph_ids,
                            "namespace": namespace,
                            "statuses": allowed_statuses,
                            "max_rank": max_rank,
                            "now": now,
                            "audiences": request.policy.audience,
                            "limit": request.topk_graph or self._config.retrieval.topk_graph,
                        },
                    ).fetchall()
                    for (memory_id,) in rows:
                        _record(memory_id, "graph", 1.0, "entity")
        return scores

    def _resolve_entity_ids(self, session, namespace: str, hints) -> list[str]:
        names = [hint.name for hint in hints]
        if not names:
            return []
        rows = session.execute(
            sa.text(
                "SELECT entity_id FROM entities "
                "WHERE namespace = :namespace AND name = ANY(:names) "
                "ORDER BY entity_id ASC"
            ),
            {"namespace": namespace, "names": names},
        ).fetchall()
        return [row[0] for row in rows]

    def _expand_graph(self, session, namespace: str, entity_ids: list[str]) -> list[str]:
        if not entity_ids or self._config.retrieval.graph_depth <= 0:
            return entity_ids
        visited = set(entity_ids)
        frontier = list(entity_ids)
        for _depth in range(self._config.retrieval.graph_depth):
            if not frontier:
                break
            rows = session.execute(
                sa.text(
                    "SELECT to_entity_id, relation, weight "
                    "FROM edges WHERE namespace = :namespace "
                    "AND from_entity_id = ANY(:entity_ids) "
                    "ORDER BY relation ASC, weight DESC, to_entity_id ASC"
                ),
                {"namespace": namespace, "entity_ids": frontier},
            ).fetchall()
            next_frontier: list[str] = []
            for to_entity_id, _relation, _weight in rows:
                if to_entity_id in visited:
                    continue
                visited.add(to_entity_id)
                next_frontier.append(to_entity_id)
                if len(visited) >= self._config.retrieval.graph_max_nodes:
                    return list(visited)
            frontier = next_frontier
        return list(visited)

    def _rank_candidates(
        self,
        candidates: dict[str, CandidateScores],
        now: dt.datetime,
        query: str,
        *,
        namespace: str | None = None,
    ) -> list[CandidateItem]:
        if not candidates:
            return []
        with self._db.session() as session:
            if namespace:
                self._set_rls_namespace(session, namespace)
            memory_ids = list(candidates.keys())
            items = self._load_memory_items(session, memory_ids)
        if not items:
            return []

        max_sem = max((candidates[item.memory_id].semantic for item in items), default=1.0)
        max_kw = max((candidates[item.memory_id].keyword for item in items), default=1.0)
        max_graph = max((candidates[item.memory_id].graph for item in items), default=1.0)

        weight = self._config.ranking
        base_scores: dict[str, float] = {}
        for item in items:
            scores = candidates[item.memory_id]
            sem = _norm(scores.semantic, max_sem)
            kw = _norm(scores.keyword, max_kw)
            graph = _norm(scores.graph, max_graph)
            recency = _recency_score(
                item.created_at, now, self._config.ranking.recency_half_life_days
            )
            base_scores[item.memory_id] = (
                weight.weight_semantic * sem
                + weight.weight_keyword * kw
                + weight.weight_graph * graph
                + weight.weight_recency * recency
                + weight.weight_importance * float(item.importance)
                + weight.weight_trust * float(item.trust_tier)
            )

        max_rerank = 1.0
        if self._config.enable_rerank and self._reranker:
            rerank_window = max(1, int(self._config.retrieval.rerank_window))
            window = sorted(
                items,
                key=lambda item: (-base_scores[item.memory_id], item.memory_id),
            )[:rerank_window]
            rerank_scores = self._reranker.score(query, [item.content_text for item in window])
            for item, score in zip(window, rerank_scores):
                entry = candidates.get(item.memory_id)
                if entry is None:
                    continue
                entry.rerank = float(score)
                entry.reasons.add("rerank")
            max_rerank = max(
                (candidates[item.memory_id].rerank for item in window), default=1.0
            )

        ranked: list[CandidateItem] = []
        for item in items:
            scores = candidates[item.memory_id]
            item.scores = scores
            rerank = _norm(scores.rerank, max_rerank)
            score = base_scores[item.memory_id]
            if self._config.enable_rerank and self._reranker:
                score += weight.weight_rerank * rerank
            item.score = score
            ranked.append(item)
        ranked.sort(key=lambda item: (-item.score, item.memory_id))
        return ranked

    def _load_memory_items(self, session, memory_ids: list[str]) -> list[CandidateItem]:
        if not memory_ids:
            return []
        sql, params = _build_in_clause("memory_id", memory_ids)
        rows = session.execute(
            sa.text(
                "SELECT memory_id, memory_type, content_text, content_json, importance, trust_tier, "
                "created_at, sensitivity_rank "
                f"FROM memory_items WHERE {sql} ORDER BY memory_id ASC"
            ),
            params,
        ).fetchall()
        items: list[CandidateItem] = []
        for row in rows:
            created_at = _ensure_aware(row[6]) or utc_now()
            items.append(
                CandidateItem(
                    memory_id=row[0],
                    memory_type=row[1],
                    content_text=row[2],
                    content_json=row[3] or {},
                    importance=float(row[4]),
                    trust_tier=float(row[5]),
                    created_at=created_at,
                    sensitivity_rank=int(row[7]),
                    scores=CandidateScores(),
                )
            )
        return items

    def _pack_cards(
        self,
        ranked: list[CandidateItem],
        request: MemoryQueryRequest,
        *,
        namespace: str | None = None,
    ) -> tuple[list[MemoryCard], bool]:
        if not ranked:
            return [], False
        max_cards = request.max_cards or self._config.retrieval.max_cards
        max_tokens = request.max_tokens or self._config.retrieval.max_tokens
        max_per_type = self._config.retrieval.max_per_type
        type_priority = self._config.retrieval.type_priority
        type_rank = {name: idx for idx, name in enumerate(type_priority)}
        ranked.sort(
            key=lambda item: (
                -item.score,
                type_rank.get(item.memory_type, len(type_rank)),
                item.memory_id,
            )
        )
        memory_ids = [item.memory_id for item in ranked]
        citations = self._load_citations(memory_ids, namespace=namespace)

        cards: list[MemoryCard] = []
        tokens = 0
        per_type: dict[str, int] = {}
        truncated = False
        for item in ranked:
            if len(cards) >= max_cards:
                truncated = True
                break
            if per_type.get(item.memory_type, 0) >= max_per_type:
                continue
            text = item.content_text
            token_count = estimate_tokens(text)
            if tokens + token_count > max_tokens:
                truncated = True
                break
            tokens += token_count
            per_type[item.memory_type] = per_type.get(item.memory_type, 0) + 1
            why = sorted(item.scores.reasons)
            cards.append(
                MemoryCard(
                    memory_id=item.memory_id,
                    memory_type=item.memory_type,
                    text=text,
                    content_json=item.content_json or {},
                    citations=citations.get(item.memory_id, []),
                    why_retrieved=why,
                    score=item.score,
                )
            )
        return cards, truncated

    def _load_citations(
        self, memory_ids: list[str], *, namespace: str | None = None
    ) -> dict[str, list[ProvenancePointer]]:
        if not memory_ids:
            return {}
        with self._db.session() as session:
            if namespace:
                self._set_rls_namespace(session, namespace)
            sql, params = _build_in_clause("memory_id", memory_ids)
            rows = session.execute(
                sa.text(
                    "SELECT memory_id, artifact_version_id, chunk_id, start_offset, end_offset, excerpt_hash "
                    f"FROM memory_provenance WHERE {sql} ORDER BY memory_id, chunk_id"
                ),
                params,
            ).fetchall()
        citations: dict[str, list[ProvenancePointer]] = {}
        for row in rows:
            pointer = ProvenancePointer(
                artifact_version_id=row[1],
                chunk_id=row[2],
                start_offset=int(row[3]),
                end_offset=int(row[4]),
                excerpt_hash=row[5],
            )
            citations.setdefault(row[0], []).append(pointer)
        return citations


def _build_in_clause(column: str, values: list[str]) -> tuple[str, dict[str, object]]:
    params: dict[str, object] = {}
    keys: list[str] = []
    for idx, value in enumerate(values):
        key = f"val_{idx}"
        params[key] = value
        keys.append(f":{key}")
    return f"{column} IN ({', '.join(keys)})", params


def _norm(value: float, max_value: float) -> float:
    if max_value <= 0.0:
        return 0.0
    return float(value) / float(max_value)


def _recency_score(created_at: dt.datetime, now: dt.datetime, half_life_days: int) -> float:
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    if half_life_days <= 0:
        return 0.0
    return math.exp(-math.log(2.0) * age_days / float(half_life_days))


def _ensure_aware(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value
