"""Lexical index utilities for retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import text

from ..logging_utils import get_logger
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
                    "USING fts5(event_id UNINDEXED, ocr_text, window_title, app_name, domain, url)"
                )
            )

    def upsert_event(self, event: EventRecord) -> None:
        engine = self._db.engine
        if engine.dialect.name == "sqlite":
            with engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM event_fts WHERE event_id = :event_id"),
                    {"event_id": event.event_id},
                )
                conn.execute(
                    text(
                        "INSERT INTO event_fts(event_id, ocr_text, window_title, app_name, domain, url) "
                        "VALUES (:event_id, :ocr_text, :window_title, :app_name, :domain, :url)"
                    ),
                    {
                        "event_id": event.event_id,
                        "ocr_text": event.ocr_text or "",
                        "window_title": event.window_title or "",
                        "app_name": event.app_name or "",
                        "domain": event.domain or "",
                        "url": event.url or "",
                    },
                )
            return

        if engine.dialect.name == "postgresql":
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE events SET "
                        "ts_end = ts_end "
                        "WHERE event_id = :event_id"
                    ),
                    {"event_id": event.event_id},
                )

    def search(self, query: str, limit: int = 20) -> list[LexicalHit]:
        engine = self._db.engine
        if not query.strip():
            return []
        if engine.dialect.name == "sqlite":
            with engine.begin() as conn:
                rows = conn.execute(
                    text(
                        "SELECT event_id, bm25(event_fts) AS rank "
                        "FROM event_fts WHERE event_fts MATCH :query "
                        "ORDER BY rank LIMIT :limit"
                    ),
                    {"query": query, "limit": limit},
                ).fetchall()
            return [
                LexicalHit(event_id=row[0], score=1 / (1 + abs(row[1])))
                for row in rows
            ]

        if engine.dialect.name == "postgresql":
            with engine.begin() as conn:
                rows = conn.execute(
                    text(
                        "SELECT event_id, ts_rank_cd("
                        "to_tsvector('english', coalesce(ocr_text,'') || ' ' || "
                        "coalesce(window_title,'') || ' ' || coalesce(app_name,'') || ' ' || "
                        "coalesce(domain,'') || ' ' || coalesce(url,'')), "
                        "plainto_tsquery('english', :query)) AS rank "
                        "FROM events "
                        "WHERE to_tsvector('english', coalesce(ocr_text,'') || ' ' || "
                        "coalesce(window_title,'') || ' ' || coalesce(app_name,'') || ' ' || "
                        "coalesce(domain,'') || ' ' || coalesce(url,'')) @@ "
                        "plainto_tsquery('english', :query) "
                        "ORDER BY rank DESC LIMIT :limit"
                    ),
                    {"query": query, "limit": limit},
                ).fetchall()
            return [LexicalHit(event_id=row[0], score=float(row[1] or 0.0)) for row in rows]

        self._log.warning("Unsupported dialect for lexical search: %s", engine.dialect.name)
        return []

    def bulk_upsert(self, events: Iterable[EventRecord]) -> None:
        for event in events:
            self.upsert_event(event)
