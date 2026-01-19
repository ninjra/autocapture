"""SQLite-backed deterministic memory store."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from ..config import MemoryConfig
from ..logging_utils import get_logger
from .models import (
    ArtifactMeta,
    MemoryGcResult,
    MemoryIngestResult,
    MemoryItemList,
    MemoryItemRecord,
    MemoryPromoteResult,
    MemoryQueryResult,
    MemorySpanHit,
    MemoryVerifyResult,
)
from .policy import DefaultPolicyEngine
from .schema import ensure_schema
from .utils import (
    EPOCH_UTC,
    coerce_timestamp,
    ensure_dir,
    format_utc,
    normalize_document_text,
    normalize_title,
    parse_iso8601,
    sanitize_fts_query,
    sha256_text,
    stable_json_dumps,
)


class MemoryStore:
    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._log = get_logger("memory.store")
        storage = config.storage
        self._root = Path(storage.root_dir).expanduser()
        self._db_path = self._root / storage.db_filename
        self._artifacts_dir = self._root / storage.artifacts_dir
        self._snapshots_dir = self._root / storage.snapshots_dir
        ensure_dir(self._root)
        ensure_dir(self._artifacts_dir)
        ensure_dir(self._snapshots_dir)
        self._fts_available = self._init_schema()
        self._policy = DefaultPolicyEngine(
            blocked_labels=config.policy.blocked_labels,
            exclude_patterns=config.policy.exclude_patterns,
            redact_patterns=config.policy.redact_patterns,
            redact_token=config.policy.redact_token,
        )

    @property
    def root_dir(self) -> Path:
        return self._root

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def artifacts_dir(self) -> Path:
        return self._artifacts_dir

    @property
    def snapshots_dir(self) -> Path:
        return self._snapshots_dir

    @property
    def fts_available(self) -> bool:
        return self._fts_available

    def ingest_text(
        self,
        text: str,
        meta: ArtifactMeta,
        *,
        timestamp: str | None = None,
    ) -> MemoryIngestResult:
        normalized = normalize_document_text(text)
        decision = self._policy.evaluate_artifact(meta, normalized)
        payload_text = normalized
        redacted = False
        if decision.action == "exclude":
            payload_text = normalized
        elif decision.action == "redact" and decision.redacted_text is not None:
            payload_text = decision.redacted_text
            redacted = True

        payload_sha256 = sha256_text(payload_text)
        meta_title = normalize_title(meta.title)
        source_uri = (meta.source_uri or "").strip()
        artifact_id = sha256_text(f"{payload_sha256}:{source_uri}:{meta_title}")
        created_at = format_utc(coerce_timestamp(timestamp, fallback=EPOCH_UTC))
        labels_json = stable_json_dumps(sorted(set(meta.labels or [])), indent=None)
        redaction_json = (
            stable_json_dumps(decision.redaction_map, indent=None)
            if decision.redaction_map
            else None
        )

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO artifacts(
                    artifact_id, source_uri, title, content_type, payload_sha256,
                    created_at, redaction_json, labels_json, excluded
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    source_uri or None,
                    meta_title or None,
                    meta.content_type or "text/plain",
                    payload_sha256,
                    created_at,
                    redaction_json,
                    labels_json,
                    1 if decision.action == "exclude" else 0,
                ),
            )

        if decision.action == "exclude":
            return MemoryIngestResult(
                artifact_id=artifact_id,
                payload_sha256=payload_sha256,
                excluded=True,
                redacted=redacted,
                warnings=["artifact_excluded"],
            )

        artifact_path = self._artifact_path(artifact_id)
        artifact_path.write_text(payload_text, encoding="utf-8")

        doc_id = artifact_id
        text_sha256 = sha256_text(payload_text)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO documents(
                    doc_id, artifact_id, title, source_uri, created_at, updated_at, labels_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    artifact_id,
                    meta_title or None,
                    source_uri or None,
                    created_at,
                    created_at,
                    labels_json,
                ),
            )
            cur.execute(
                """
                INSERT OR REPLACE INTO document_text(doc_id, text, text_sha256)
                VALUES (?, ?, ?)
                """,
                (doc_id, payload_text, text_sha256),
            )
            cur.execute("DELETE FROM spans WHERE doc_id = ?", (doc_id,))
            if self._fts_available:
                cur.execute("DELETE FROM spans_fts WHERE doc_id = ?", (doc_id,))

        spans = list(
            _chunk_text(
                payload_text,
                max_chars=self._config.spans.max_chars,
                min_chars=self._config.spans.min_chars,
            )
        )
        span_ids: list[str] = []
        with self._connect() as conn:
            cur = conn.cursor()
            for start, end, section_path, span_text in spans:
                span_decision = self._policy.evaluate_span(meta, span_text)
                if span_decision.action == "exclude":
                    continue
                if span_decision.action == "redact" and span_decision.redacted_text is not None:
                    span_text = span_decision.redacted_text
                span_sha256 = sha256_text(span_text)
                span_id = sha256_text(f"{doc_id}:{start}:{end}:{span_sha256}")
                span_ids.append(span_id)
                cur.execute(
                    """
                    INSERT OR REPLACE INTO spans(
                        span_id, doc_id, start, end, section_path, text, span_sha256,
                        labels_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        span_id,
                        doc_id,
                        int(start),
                        int(end),
                        section_path,
                        span_text,
                        span_sha256,
                        None,
                        created_at,
                    ),
                )
                if self._fts_available:
                    cur.execute(
                        "DELETE FROM spans_fts WHERE span_id = ?",
                        (span_id,),
                    )
                    cur.execute(
                        """
                        INSERT INTO spans_fts(span_id, doc_id, title, section_path, text)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            span_id,
                            doc_id,
                            meta_title or "",
                            section_path or "",
                            span_text,
                        ),
                    )

        return MemoryIngestResult(
            artifact_id=artifact_id,
            doc_id=doc_id,
            payload_sha256=payload_sha256,
            span_ids=span_ids,
            span_count=len(span_ids),
            excluded=False,
            redacted=redacted,
        )

    def query_spans(self, query: str, *, k: int | None = None) -> MemoryQueryResult:
        query = (query or "").strip()
        if not query:
            return MemoryQueryResult(spans=[], retrieval_disabled=False, reason="query_empty")
        if not self._config.retrieval.enabled:
            return MemoryQueryResult(spans=[], retrieval_disabled=True, reason="retrieval_disabled")
        if not self._fts_available:
            return MemoryQueryResult(spans=[], retrieval_disabled=True, reason="fts_unavailable")
        k = int(k or self._config.retrieval.default_k)
        k = max(1, min(k, self._config.retrieval.max_k))
        sanitized = sanitize_fts_query(query)
        if not sanitized:
            return MemoryQueryResult(spans=[], retrieval_disabled=False, reason="query_empty")

        rows: list[sqlite3.Row] = []
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT spans.span_id,
                           spans.doc_id,
                           spans.start,
                           spans.end,
                           spans.section_path,
                           spans.text,
                           spans.span_sha256,
                           documents.title,
                           documents.source_uri,
                           documents.updated_at,
                           document_text.text_sha256,
                           bm25(spans_fts) AS rank
                    FROM spans_fts
                    JOIN spans ON spans.span_id = spans_fts.span_id
                    JOIN documents ON documents.doc_id = spans.doc_id
                    JOIN document_text ON document_text.doc_id = spans.doc_id
                    WHERE spans_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (sanitized, k * 4),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if sanitized != query:
                    try:
                        rows = conn.execute(
                            """
                            SELECT spans.span_id,
                                   spans.doc_id,
                                   spans.start,
                                   spans.end,
                                   spans.section_path,
                                   spans.text,
                                   spans.span_sha256,
                                   documents.title,
                                   documents.source_uri,
                                   documents.updated_at,
                                   document_text.text_sha256,
                                   bm25(spans_fts) AS rank
                            FROM spans_fts
                            JOIN spans ON spans.span_id = spans_fts.span_id
                            JOIN documents ON documents.doc_id = spans.doc_id
                            JOIN document_text ON document_text.doc_id = spans.doc_id
                            WHERE spans_fts MATCH ?
                            ORDER BY rank
                            LIMIT ?
                            """,
                            (sanitized, k * 4),
                        ).fetchall()
                    except sqlite3.OperationalError as exc_inner:
                        self._log.warning("FTS query failed: {}", exc_inner)
                        return MemoryQueryResult(
                            spans=[], retrieval_disabled=False, reason="query_failed"
                        )
                else:
                    self._log.warning("FTS query failed: {}", exc)
                    return MemoryQueryResult(
                        spans=[], retrieval_disabled=False, reason="query_failed"
                    )

        if not rows:
            return MemoryQueryResult(spans=[], retrieval_disabled=False, reason="no_results")

        ref_ts = _max_updated_at(rows)
        half_life = max(1, int(self._config.retrieval.recency_half_life_days))
        spans: list[MemorySpanHit] = []
        for row in rows:
            bm25_rank = float(row["rank"]) if row["rank"] is not None else 0.0
            bm25_score = 1.0 / (1.0 + abs(bm25_rank))
            updated_at = parse_iso8601(row["updated_at"]) or EPOCH_UTC
            age_days = (ref_ts - updated_at).total_seconds() / 86400.0
            recency = pow(2.0, -(age_days / half_life))
            score = 0.85 * bm25_score + 0.15 * recency
            spans.append(
                MemorySpanHit(
                    span_id=row["span_id"],
                    doc_id=row["doc_id"],
                    start=int(row["start"]),
                    end=int(row["end"]),
                    section_path=row["section_path"],
                    text=row["text"],
                    span_sha256=row["span_sha256"],
                    title=row["title"],
                    source_uri=row["source_uri"],
                    text_sha256=row["text_sha256"],
                    score=round(score, 6),
                )
            )

        deduped: dict[str, MemorySpanHit] = {}
        for span in spans:
            existing = deduped.get(span.span_sha256)
            if existing is None:
                deduped[span.span_sha256] = span
                continue
            if span.score > existing.score:
                deduped[span.span_sha256] = span
            elif span.score == existing.score and span.span_id < existing.span_id:
                deduped[span.span_sha256] = span

        ranked = sorted(deduped.values(), key=lambda item: (-item.score, item.span_id))
        return MemoryQueryResult(spans=ranked[:k])

    def list_items(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> MemoryItemList:
        status = (status or "").strip().lower()
        params: list[object] = []
        clause = ""
        if status and status != "all":
            clause = "WHERE status = ?"
            params.append(status)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT item_id, key, value, item_type, status, tags_json,
                       created_at, updated_at, value_sha256, user_asserted
                FROM memory_items
                {clause}
                ORDER BY key ASC, item_id ASC
                LIMIT ? OFFSET ?
                """.format(
                    clause=clause
                ),
                (*params, limit, offset),
            ).fetchall()
        items = [
            MemoryItemRecord(
                item_id=row[0],
                key=row[1],
                value=row[2],
                item_type=row[3],
                status=row[4],
                tags=json.loads(row[5]) if row[5] else [],
                created_at=row[6],
                updated_at=row[7],
                value_sha256=row[8],
                user_asserted=bool(row[9]),
            )
            for row in rows
        ]
        return MemoryItemList(items=items)

    def latest_span_timestamp(self, span_ids: Iterable[str]) -> str:
        ids = [sid for sid in span_ids if sid]
        if not ids:
            return format_utc(EPOCH_UTC)
        placeholders = ",".join(["?"] * len(ids))
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT MAX(documents.updated_at) AS latest
                FROM spans
                JOIN documents ON documents.doc_id = spans.doc_id
                WHERE spans.span_id IN ({placeholders})
                """,
                ids,
            ).fetchone()
        parsed = parse_iso8601(row["latest"]) if row and row["latest"] else None
        return format_utc(parsed or EPOCH_UTC)

    def record_snapshot(
        self,
        *,
        snapshot_id: str,
        query: str,
        created_at: str,
        config_sha256: str,
        output_sha256: str,
        retrieval_disabled: bool,
        span_ids: Iterable[str],
        item_ids: Iterable[str],
    ) -> None:
        span_ids = [sid for sid in span_ids if sid]
        item_ids = [iid for iid in item_ids if iid]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO context_snapshots(
                    snapshot_id, query, created_at, config_sha256, output_sha256,
                    retrieval_disabled, span_count, item_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    query,
                    created_at,
                    config_sha256,
                    output_sha256,
                    1 if retrieval_disabled else 0,
                    len(span_ids),
                    len(item_ids),
                ),
            )
            conn.execute(
                "DELETE FROM snapshot_spans WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            for rank, span_id in enumerate(span_ids, start=1):
                conn.execute(
                    "INSERT INTO snapshot_spans(snapshot_id, span_id, rank) VALUES (?, ?, ?)",
                    (snapshot_id, span_id, rank),
                )

    def propose_item(
        self,
        *,
        key: str,
        value: str,
        item_type: str,
        tags: Iterable[str] | None = None,
        span_ids: Iterable[str] | None = None,
        user_asserted: bool = False,
        timestamp: str | None = None,
    ) -> MemoryItemRecord:
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError("key and value are required")
        item_type = item_type.strip() or "fact"
        created_at = format_utc(coerce_timestamp(timestamp, fallback=EPOCH_UTC))
        value_sha256 = sha256_text(value)
        item_id = sha256_text(f"{key}:{value}:{item_type}:{value_sha256}")
        tags_list = sorted({tag.strip() for tag in (tags or []) if tag})
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_items(
                    item_id, key, value, item_type, status, tags_json,
                    created_at, updated_at, value_sha256, user_asserted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    key,
                    value,
                    item_type,
                    "proposed",
                    stable_json_dumps(tags_list, indent=None),
                    created_at,
                    created_at,
                    value_sha256,
                    1 if user_asserted else 0,
                ),
            )
            _store_item_sources(conn, item_id, span_ids)
        return MemoryItemRecord(
            item_id=item_id,
            key=key,
            value=value,
            item_type=item_type,
            status="proposed",
            tags=tags_list,
            created_at=created_at,
            updated_at=created_at,
            value_sha256=value_sha256,
            user_asserted=user_asserted,
        )

    def promote_item(
        self,
        *,
        item_id: str,
        span_ids: Iterable[str] | None = None,
        user_asserted: bool = False,
        timestamp: str | None = None,
    ) -> MemoryPromoteResult:
        updated_at = format_utc(coerce_timestamp(timestamp, fallback=EPOCH_UTC))
        deprecated: list[str] = []
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT item_id, key, status, user_asserted
                FROM memory_items
                WHERE item_id = ?
                """,
                (item_id,),
            ).fetchone()
            if not row:
                raise ValueError("memory item not found")
            key = row[1]
            existing_asserted = bool(row[3])
            sources = _get_item_sources(conn, item_id)
            sources = set(sources)
            if span_ids:
                sources.update({sid for sid in span_ids if sid})
            if not sources and not (user_asserted or existing_asserted):
                raise ValueError("promotion requires span_id(s) or --user-asserted")
            _store_item_sources(conn, item_id, sources)
            active_rows = conn.execute(
                """
                SELECT item_id FROM memory_items
                WHERE key = ? AND status = 'active' AND item_id != ?
                """,
                (key, item_id),
            ).fetchall()
            deprecated = [row[0] for row in active_rows]
            for dep_id in deprecated:
                conn.execute(
                    "UPDATE memory_items SET status = 'deprecated' WHERE item_id = ?",
                    (dep_id,),
                )
            conn.execute(
                """
                UPDATE memory_items
                SET status = 'active', updated_at = ?, user_asserted = ?
                WHERE item_id = ?
                """,
                (updated_at, 1 if (user_asserted or existing_asserted) else 0, item_id),
            )
        return MemoryPromoteResult(item_id=item_id, status="active", deprecated_item_ids=deprecated)

    def verify(self) -> MemoryVerifyResult:
        errors: list[str] = []
        with self._connect() as conn:
            artifacts = conn.execute(
                "SELECT artifact_id, payload_sha256, excluded FROM artifacts"
            ).fetchall()
            for artifact_id, payload_sha256, excluded in artifacts:
                if int(excluded or 0) == 1:
                    continue
                path = self._artifact_path(artifact_id)
                if not path.exists():
                    errors.append(f"missing artifact payload: {artifact_id}")
                    continue
                data = path.read_text(encoding="utf-8")
                if sha256_text(data) != payload_sha256:
                    errors.append(f"artifact checksum mismatch: {artifact_id}")

            doc_rows = conn.execute(
                "SELECT doc_id, text, text_sha256 FROM document_text"
            ).fetchall()
            doc_text_map = {row[0]: row[1] for row in doc_rows}
            for doc_id, text, text_sha256 in doc_rows:
                if sha256_text(text) != text_sha256:
                    errors.append(f"document_text checksum mismatch: {doc_id}")

            span_rows = conn.execute(
                "SELECT span_id, doc_id, start, end, text, span_sha256 FROM spans"
            ).fetchall()
            for span_id, doc_id, start, end, span_text, span_sha256 in span_rows:
                doc_text = doc_text_map.get(doc_id)
                if doc_text is None:
                    errors.append(f"span references missing doc: {span_id}")
                    continue
                sliced = doc_text[int(start) : int(end)]
                if sliced != span_text:
                    errors.append(f"span text mismatch: {span_id}")
                    continue
                if sha256_text(span_text) != span_sha256:
                    errors.append(f"span checksum mismatch: {span_id}")

            snapshots = conn.execute(
                "SELECT snapshot_id, output_sha256 FROM context_snapshots"
            ).fetchall()
            for snapshot_id, output_sha256 in snapshots:
                snapshot_dir = self._snapshots_dir / snapshot_id
                context_path = snapshot_dir / "context.md"
                citations_path = snapshot_dir / "citations.json"
                manifest_path = snapshot_dir / "context.json"
                hash_path = snapshot_dir / "snapshot.hash"
                if not context_path.exists() or not citations_path.exists():
                    errors.append(f"missing snapshot files: {snapshot_id}")
                    continue
                output_hash = sha256_text(
                    context_path.read_text(encoding="utf-8")
                    + "\n"
                    + citations_path.read_text(encoding="utf-8")
                )
                if output_hash != output_sha256:
                    errors.append(f"snapshot checksum mismatch: {snapshot_id}")
                if manifest_path.exists():
                    try:
                        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        errors.append(f"context.json invalid: {snapshot_id}")
                    else:
                        if payload.get("output_sha256") != output_sha256:
                            errors.append(f"context.json hash mismatch: {snapshot_id}")
                if hash_path.exists():
                    stored = hash_path.read_text(encoding="utf-8").strip()
                    if stored != output_sha256:
                        errors.append(f"snapshot.hash mismatch: {snapshot_id}")
        return MemoryVerifyResult(ok=not errors, errors=errors)

    def gc_snapshots(self, *, retention_days: int | None = None) -> MemoryGcResult:
        retention_days = retention_days or self._config.storage.snapshot_retention_days
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=int(retention_days))
        removed_snapshots = 0
        removed_dirs = 0
        with self._connect() as conn:
            rows = conn.execute("SELECT snapshot_id, created_at FROM context_snapshots").fetchall()
            to_remove = []
            for snapshot_id, created_at in rows:
                parsed = parse_iso8601(created_at)
                if parsed is None:
                    continue
                if parsed <= cutoff:
                    to_remove.append(snapshot_id)
            for snapshot_id in to_remove:
                conn.execute(
                    "DELETE FROM context_snapshots WHERE snapshot_id = ?",
                    (snapshot_id,),
                )
                removed_snapshots += 1
                snapshot_dir = self._snapshots_dir / snapshot_id
                if snapshot_dir.exists():
                    for child in snapshot_dir.iterdir():
                        if child.is_file():
                            child.unlink(missing_ok=True)
                    try:
                        snapshot_dir.rmdir()
                        removed_dirs += 1
                    except OSError:
                        pass
        return MemoryGcResult(removed_snapshots=removed_snapshots, removed_dirs=removed_dirs)

    @contextmanager
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> bool:
        with self._connect() as conn:
            try:
                return ensure_schema(conn, require_fts=self._config.storage.require_fts)
            except sqlite3.OperationalError as exc:
                self._log.warning("Failed to initialize FTS5: {}", exc)
                if self._config.storage.require_fts:
                    raise
                return False

    def _artifact_path(self, artifact_id: str) -> Path:
        return self._artifacts_dir / f"{artifact_id}.txt"


def _chunk_text(
    text: str,
    *,
    max_chars: int,
    min_chars: int,
) -> Iterable[tuple[int, int, str | None, str]]:
    if not text:
        return []
    max_chars = max(1, int(max_chars))
    min_chars = max(0, int(min_chars))
    idx = 0
    length = len(text)
    section_path: str | None = None
    while idx < length:
        if text[idx] == "#":
            line_end = text.find("\n", idx)
            if line_end == -1:
                line_end = length
            section_path = text[idx:line_end].strip()
        end = min(idx + max_chars, length)
        split = text.rfind("\n\n", idx + min_chars, end)
        if split != -1:
            end = split
        span_text = text[idx:end]
        if span_text.strip():
            yield idx, end, section_path, span_text
        idx = end
        while idx < length and text[idx] == "\n":
            idx += 1


def _store_item_sources(
    conn: sqlite3.Connection, item_id: str, span_ids: Iterable[str] | None
) -> None:
    if not span_ids:
        return
    unique_ids = sorted({sid for sid in span_ids if sid})
    for span_id in unique_ids:
        conn.execute(
            "INSERT OR IGNORE INTO memory_item_sources(item_id, span_id) VALUES (?, ?)",
            (item_id, span_id),
        )


def _get_item_sources(conn: sqlite3.Connection, item_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT span_id FROM memory_item_sources WHERE item_id = ?",
        (item_id,),
    ).fetchall()
    return [row[0] for row in rows]


def _max_updated_at(rows: Iterable[sqlite3.Row]) -> dt.datetime:
    latest = EPOCH_UTC
    for row in rows:
        parsed = parse_iso8601(row["updated_at"])
        if parsed and parsed > latest:
            latest = parsed
    return latest
