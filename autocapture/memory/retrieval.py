"""Local retrieval utilities for events and evidence."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import select

from ..storage.database import DatabaseManager
from ..storage.models import EventRecord


@dataclass(frozen=True)
class RetrieveFilters:
    apps: list[str] | None = None
    domains: list[str] | None = None


@dataclass(frozen=True)
class RetrievedEvent:
    event: EventRecord
    score: float


class RetrievalService:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def retrieve(
        self,
        query: str,
        time_range: tuple[dt.datetime, dt.datetime] | None,
        filters: RetrieveFilters | None,
        limit: int = 12,
    ) -> list[RetrievedEvent]:
        if not query.strip():
            return []
        with self._db.session() as session:
            stmt = select(EventRecord).where(EventRecord.ocr_text.ilike(f"%{query}%"))
            if time_range:
                stmt = stmt.where(EventRecord.ts_start.between(*time_range))
            if filters and filters.apps:
                stmt = stmt.where(EventRecord.app_name.in_(filters.apps))
            if filters and filters.domains:
                stmt = stmt.where(EventRecord.domain.in_(filters.domains))
            stmt = stmt.order_by(EventRecord.ts_start.desc()).limit(limit)
            rows = session.execute(stmt).scalars().all()
        results: list[RetrievedEvent] = []
        for event in rows:
            score = min(0.99, 0.5 + len(query) / max(len(event.ocr_text), 1))
            results.append(RetrievedEvent(event=event, score=score))
        return results

    def list_events(self, limit: int = 100) -> Iterable[EventRecord]:
        with self._db.session() as session:
            stmt = select(EventRecord).order_by(EventRecord.ts_start.desc()).limit(limit)
            return list(session.execute(stmt).scalars().all())
