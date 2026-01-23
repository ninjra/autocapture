"""Lexical index utilities for retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import re

from sqlalchemy import bindparam, text
from sqlalchemy.exc import OperationalError

from ..logging_utils import get_logger
from ..text.normalize import normalize_text
from ..storage.database import DatabaseManager
from ..storage.sqlite_features import sqlite_fts5_available
from ..storage.models import EventRecord


@dataclass(frozen=True)
class LexicalHit:
    event_id: str
    score: float


@dataclass(frozen=True)
class SpanLexicalHit:
    span_id: str
    score: float


class LexicalIndex:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._log = get_logger("index.lexical")
        self._fts_available = sqlite_fts5_available(self._db.engine)
        if not self._fts_available:
            self._log.warning("SQLite FTS5 unavailable; lexical search will use LIKE fallback.")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        engine = self._db.engine
        if engine.dialect.name != "sqlite" or not self._fts_available:
            return
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS event_fts "
                    "USING fts5(event_id UNINDEXED, ocr_text, window_title, app_name, domain, url, agent_text)"
                )
            )
            try:
                conn.execute(text("ALTER TABLE event_fts ADD COLUMN agent_text"))
            except Exception:
                pass

    def upsert_event(self, event: EventRecord) -> None:
        engine = self._db.engine
        if engine.dialect.name == "sqlite":
            if not self._fts_available:
                return
            with engine.begin() as conn:
                existing_agent = conn.execute(
                    text("SELECT agent_text FROM event_fts WHERE event_id = :event_id"),
                    {"event_id": event.event_id},
                ).scalar()
                agent_text = existing_agent or ""
                ocr_text = event.ocr_text_normalized or event.ocr_text or ""
                layout_md = _extract_layout_md(event.tags)
                if layout_md:
                    if event.ocr_text_normalized:
                        layout_md = normalize_text(layout_md)
                    ocr_text = f"{ocr_text}\n\n{layout_md}".strip()
                conn.execute(
                    text("DELETE FROM event_fts WHERE event_id = :event_id"),
                    {"event_id": event.event_id},
                )
                conn.execute(
                    text(
                        "INSERT INTO event_fts("
                        "event_id, ocr_text, window_title, app_name, domain, url, agent_text"
                        ") VALUES ("
                        ":event_id, :ocr_text, :window_title, :app_name, :domain, :url, :agent_text"
                        ")"
                    ),
                    {
                        "event_id": event.event_id,
                        "ocr_text": ocr_text,
                        "window_title": event.window_title or "",
                        "app_name": event.app_name or "",
                        "domain": event.domain or "",
                        "url": event.url or "",
                        "agent_text": agent_text,
                    },
                )
            return

        if engine.dialect.name == "postgresql":
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE events SET " "ts_end = ts_end " "WHERE event_id = :event_id"),
                    {"event_id": event.event_id},
                )

    def bulk_upsert(self, events: Iterable[EventRecord]) -> None:
        for event in events:
            self.upsert_event(event)

    def delete_events(self, event_ids: Iterable[str]) -> int:
        ids = [str(item) for item in event_ids if item]
        if not ids:
            return 0
        engine = self._db.engine
        if engine.dialect.name != "sqlite" or not self._fts_available:
            return 0
        deleted = 0
        with engine.begin() as conn:
            stmt = text("DELETE FROM event_fts WHERE event_id IN :event_ids").bindparams(
                bindparam("event_ids", expanding=True)
            )
            result = conn.execute(stmt, {"event_ids": ids})
            deleted = int(result.rowcount or 0)
        return deleted

    def upsert_agent_text(self, event_id: str, agent_text: str) -> None:
        engine = self._db.engine
        if engine.dialect.name != "sqlite" or not self._fts_available:
            return
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE event_fts SET agent_text = :agent_text WHERE event_id = :event_id"),
                {"agent_text": agent_text, "event_id": event_id},
            )

    def search(self, query: str, limit: int = 20) -> list[LexicalHit]:
        engine = self._db.engine
        if not query.strip():
            return []
        if engine.dialect.name == "sqlite":
            if not self._fts_available:
                return _fallback_event_search(engine, query, limit)
            rows = []
            with engine.begin() as conn:
                try:
                    rows = conn.execute(
                        text(
                            "SELECT event_id, bm25(event_fts) AS rank "
                            "FROM event_fts WHERE event_fts MATCH :query "
                            "ORDER BY rank LIMIT :limit"
                        ),
                        {"query": query, "limit": limit},
                    ).fetchall()
                except OperationalError:
                    sanitized = _sanitize_fts_query(query)
                    if not sanitized:
                        return []
                    rows = conn.execute(
                        text(
                            "SELECT event_id, bm25(event_fts) AS rank "
                            "FROM event_fts WHERE event_fts MATCH :query "
                            "ORDER BY rank LIMIT :limit"
                        ),
                        {"query": sanitized, "limit": limit},
                    ).fetchall()
            return [LexicalHit(event_id=row[0], score=1 / (1 + abs(row[1]))) for row in rows]

        if engine.dialect.name == "postgresql":
            with engine.begin() as conn:
                rows = conn.execute(
                    text(
                        "SELECT event_id, ts_rank_cd("
                        "to_tsvector('english', coalesce(ocr_text_normalized, ocr_text,'') || ' ' || "
                        "coalesce(window_title,'') || ' ' || coalesce(app_name,'') || ' ' || "
                        "coalesce(domain,'') || ' ' || coalesce(url,'')), "
                        "plainto_tsquery('english', :query)) AS rank "
                        "FROM events "
                        "WHERE to_tsvector('english', coalesce(ocr_text_normalized, ocr_text,'') || ' ' || "
                        "coalesce(window_title,'') || ' ' || coalesce(app_name,'') || ' ' || "
                        "coalesce(domain,'') || ' ' || coalesce(url,'')) @@ "
                        "plainto_tsquery('english', :query) "
                        "ORDER BY rank DESC LIMIT :limit"
                    ),
                    {"query": query, "limit": limit},
                ).fetchall()
            return [LexicalHit(event_id=row[0], score=float(row[1] or 0.0)) for row in rows]

        self._log.warning("Unsupported dialect for lexical search: {}", engine.dialect.name)
        return []


class SpanLexicalIndex:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._log = get_logger("index.lexical.spans")
        self._fts_available = sqlite_fts5_available(self._db.engine)
        if not self._fts_available:
            self._log.warning("SQLite FTS5 unavailable; span search will use LIKE fallback.")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        engine = self._db.engine
        if engine.dialect.name != "sqlite" or not self._fts_available:
            return
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS span_fts "
                    "USING fts5(span_id UNINDEXED, event_id UNINDEXED, frame_id UNINDEXED, text)"
                )
            )
            conn.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                    "USING fts5(chunk_id UNINDEXED, event_id UNINDEXED, text)"
                )
            )

    def upsert_span(self, *, span_id: str, event_id: str, frame_id: str, text: str) -> None:
        if not span_id:
            return
        engine = self._db.engine
        if engine.dialect.name != "sqlite" or not self._fts_available:
            return
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM span_fts WHERE span_id = :span_id"),
                {"span_id": span_id},
            )
            conn.execute(
                text(
                    "INSERT INTO span_fts(span_id, event_id, frame_id, text) "
                    "VALUES (:span_id, :event_id, :frame_id, :text)"
                ),
                {
                    "span_id": span_id,
                    "event_id": event_id,
                    "frame_id": frame_id,
                    "text": text or "",
                },
            )
            conn.execute(
                text("DELETE FROM chunks_fts WHERE chunk_id = :chunk_id"),
                {"chunk_id": span_id},
            )
            conn.execute(
                text(
                    "INSERT INTO chunks_fts(chunk_id, event_id, text) "
                    "VALUES (:chunk_id, :event_id, :text)"
                ),
                {
                    "chunk_id": span_id,
                    "event_id": event_id,
                    "text": text or "",
                },
            )

    def bulk_upsert(self, spans: Iterable[dict]) -> None:
        for item in spans:
            self.upsert_span(
                span_id=str(item.get("span_id") or ""),
                event_id=str(item.get("event_id") or ""),
                frame_id=str(item.get("frame_id") or ""),
                text=str(item.get("text") or ""),
            )

    def search(self, query: str, limit: int = 20) -> list[SpanLexicalHit]:
        engine = self._db.engine
        if not query.strip():
            return []
        if engine.dialect.name == "sqlite":
            if not self._fts_available:
                return _fallback_span_search(engine, query, limit)
            rows = []
            with engine.begin() as conn:
                try:
                    rows = conn.execute(
                        text(
                            "SELECT chunk_id, bm25(chunks_fts) AS rank "
                            "FROM chunks_fts WHERE chunks_fts MATCH :query "
                            "ORDER BY rank LIMIT :limit"
                        ),
                        {"query": query, "limit": limit},
                    ).fetchall()
                except OperationalError:
                    sanitized = _sanitize_fts_query(query)
                    if not sanitized:
                        return []
                    rows = conn.execute(
                        text(
                            "SELECT chunk_id, bm25(chunks_fts) AS rank "
                            "FROM chunks_fts WHERE chunks_fts MATCH :query "
                            "ORDER BY rank LIMIT :limit"
                        ),
                        {"query": sanitized, "limit": limit},
                    ).fetchall()
            return [SpanLexicalHit(span_id=row[0], score=1 / (1 + abs(row[1]))) for row in rows]
        self._log.warning("Unsupported dialect for span lexical search: {}", engine.dialect.name)
        return []


def _sanitize_fts_query(query: str) -> str:
    tokens = re.findall(r"[\w\.-]+", query)
    if not tokens:
        return ""
    # Avoid f-string backslash escapes inside the expression part.
    # SQLite FTS5 requires embedded quotes to be escaped by doubling them.
    quoted = ['"' + token.replace('"', '""') + '"' for token in tokens]
    return " AND ".join(quoted)


def _extract_layout_md(tags: object) -> str:
    if not isinstance(tags, dict):
        return ""
    value = tags.get("layout_md")
    if isinstance(value, str):
        return value.strip()
    return ""


def _like_pattern(query: str) -> str:
    normalized = normalize_text(query).strip() if query else ""
    base = normalized or query.strip()
    if not base:
        return ""
    escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _fallback_event_search(engine, query: str, limit: int) -> list[LexicalHit]:
    pattern = _like_pattern(query)
    if not pattern:
        return []
    sql = (
        "SELECT event_id FROM events WHERE "
        "lower(coalesce(ocr_text_normalized, ocr_text, '')) LIKE lower(:pattern) ESCAPE '\\' "
        "OR lower(coalesce(window_title, '')) LIKE lower(:pattern) ESCAPE '\\' "
        "OR lower(coalesce(app_name, '')) LIKE lower(:pattern) ESCAPE '\\' "
        "OR lower(coalesce(domain, '')) LIKE lower(:pattern) ESCAPE '\\' "
        "OR lower(coalesce(url, '')) LIKE lower(:pattern) ESCAPE '\\' "
        "LIMIT :limit"
    )
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"pattern": pattern, "limit": limit}).fetchall()
    return [LexicalHit(event_id=row[0], score=1.0) for row in rows]


def _fallback_span_search(engine, query: str, limit: int) -> list[SpanLexicalHit]:
    pattern = _like_pattern(query)
    if not pattern:
        return []
    sql = (
        "SELECT span_id FROM citable_spans "
        "WHERE lower(coalesce(text, '')) LIKE lower(:pattern) ESCAPE '\\' "
        "LIMIT :limit"
    )
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"pattern": pattern, "limit": limit}).fetchall()
    return [SpanLexicalHit(span_id=row[0], score=1.0) for row in rows]
