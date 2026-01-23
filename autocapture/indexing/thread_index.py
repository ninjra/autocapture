"""Thread summary lexical indexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.sqlite_features import sqlite_fts5_available


@dataclass(frozen=True)
class ThreadLexicalHit:
    thread_id: str
    score: float


class ThreadLexicalIndex:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._log = get_logger("index.thread.lexical")
        self._fts_available = sqlite_fts5_available(self._db.engine)
        if not self._fts_available:
            self._log.warning("SQLite FTS5 unavailable; thread search will use LIKE fallback.")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        engine = self._db.engine
        if engine.dialect.name != "sqlite" or not self._fts_available:
            return
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS thread_fts "
                    "USING fts5(thread_id UNINDEXED, title, summary, entities, tasks)"
                )
            )

    def upsert_thread(
        self,
        *,
        thread_id: str,
        title: str,
        summary: str,
        entities: Iterable[str],
        tasks: Iterable[str],
    ) -> None:
        engine = self._db.engine
        if engine.dialect.name != "sqlite" or not self._fts_available:
            return
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM thread_fts WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
            conn.execute(
                text(
                    "INSERT INTO thread_fts(thread_id, title, summary, entities, tasks) "
                    "VALUES (:thread_id, :title, :summary, :entities, :tasks)"
                ),
                {
                    "thread_id": thread_id,
                    "title": title or "",
                    "summary": summary or "",
                    "entities": " ".join(str(item) for item in entities if item),
                    "tasks": " ".join(str(item) for item in tasks if item),
                },
            )

    def search(self, query: str, limit: int = 20) -> list[ThreadLexicalHit]:
        engine = self._db.engine
        if not query.strip():
            return []
        if engine.dialect.name != "sqlite":
            return []
        if not self._fts_available:
            return _fallback_thread_search(engine, query, limit)
        with engine.begin() as conn:
            try:
                rows = conn.execute(
                    text(
                        "SELECT thread_id, bm25(thread_fts) AS rank "
                        "FROM thread_fts WHERE thread_fts MATCH :query "
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
                        "SELECT thread_id, bm25(thread_fts) AS rank "
                        "FROM thread_fts WHERE thread_fts MATCH :query "
                        "ORDER BY rank LIMIT :limit"
                    ),
                    {"query": sanitized, "limit": limit},
                ).fetchall()
        return [ThreadLexicalHit(thread_id=row[0], score=1 / (1 + abs(row[1]))) for row in rows]


def _sanitize_fts_query(query: str) -> str:
    tokens = [token for token in query.split() if token.strip()]
    if not tokens:
        return ""
    quoted = ['"' + token.replace('"', '""') + '"' for token in tokens]
    return " AND ".join(quoted)


def _like_pattern(query: str) -> str:
    base = query.strip()
    if not base:
        return ""
    escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _fallback_thread_search(engine, query: str, limit: int) -> list[ThreadLexicalHit]:
    pattern = _like_pattern(query)
    if not pattern:
        return []
    sql = (
        "SELECT thread_id FROM threads "
        "WHERE lower(coalesce(app_name, '')) LIKE lower(:pattern) ESCAPE '\\' "
        "OR lower(coalesce(window_title, '')) LIKE lower(:pattern) ESCAPE '\\' "
        "LIMIT :limit"
    )
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"pattern": pattern, "limit": limit}).fetchall()
    return [ThreadLexicalHit(thread_id=row[0], score=1.0) for row in rows]
