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
from ..storage.models import EventRecord


@dataclass(frozen=True)
class LexicalHit:
    event_id: str
    score: float


class LexicalIndex:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._log = get_logger("index.lexical")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        engine = self._db.engine
        if engine.dialect.name != "sqlite":
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
        if engine.dialect.name != "sqlite":
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
        if engine.dialect.name != "sqlite":
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
