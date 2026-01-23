"""Event query helpers for UX/event browsing."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from sqlalchemy import and_, func, or_, select

from ..database import DatabaseManager
from ..models import EventRecord, OCRSpanRecord


@dataclass(frozen=True)
class EventCursor:
    ts_start: dt.datetime
    event_id: str
    direction: str = "desc"


@dataclass(frozen=True)
class EventListFilters:
    start_utc: dt.datetime | None = None
    end_utc: dt.datetime | None = None
    apps: Sequence[str] = ()
    domains: Sequence[str] = ()
    process: str | None = None
    window_title: str | None = None
    has_screenshot: bool | None = None
    has_focus: bool | None = None
    event_ids: Sequence[str] = ()


@dataclass(frozen=True)
class FacetCount:
    value: str
    count: int


def list_events(
    db: DatabaseManager,
    *,
    filters: EventListFilters,
    cursor: EventCursor | None,
    limit: int,
) -> list[EventRecord]:
    stmt = select(EventRecord)
    stmt = _apply_filters(stmt, filters)
    stmt = _apply_cursor(stmt, cursor)
    if cursor and cursor.direction == "asc":
        stmt = stmt.order_by(EventRecord.ts_start.asc(), EventRecord.event_id.asc())
    else:
        stmt = stmt.order_by(EventRecord.ts_start.desc(), EventRecord.event_id.desc())
    stmt = stmt.limit(limit)
    with db.session() as session:
        return session.execute(stmt).scalars().all()


def get_event(db: DatabaseManager, event_id: str) -> EventRecord | None:
    with db.session() as session:
        return session.get(EventRecord, event_id)


def list_facets(
    db: DatabaseManager,
    *,
    filters: EventListFilters,
    app_limit: int = 50,
    domain_limit: int = 50,
) -> tuple[list[FacetCount], list[FacetCount]]:
    apps_stmt = select(EventRecord.app_name.label("value"), func.count().label("count"))
    apps_stmt = apps_stmt.select_from(EventRecord)
    apps_stmt = _apply_filters(apps_stmt, filters)
    apps_stmt = apps_stmt.group_by(EventRecord.app_name)
    apps_stmt = apps_stmt.order_by(func.count().desc(), EventRecord.app_name.asc())
    apps_stmt = apps_stmt.limit(app_limit)

    domains_stmt = select(EventRecord.domain.label("value"), func.count().label("count"))
    domains_stmt = domains_stmt.select_from(EventRecord)
    domains_stmt = _apply_filters(domains_stmt, filters)
    domains_stmt = domains_stmt.where(EventRecord.domain.is_not(None))
    domains_stmt = domains_stmt.where(EventRecord.domain != "")
    domains_stmt = domains_stmt.group_by(EventRecord.domain)
    domains_stmt = domains_stmt.order_by(func.count().desc(), EventRecord.domain.asc())
    domains_stmt = domains_stmt.limit(domain_limit)

    with db.session() as session:
        app_rows = session.execute(apps_stmt).all()
        domain_rows = session.execute(domains_stmt).all()

    apps = [FacetCount(value=row[0], count=int(row[1] or 0)) for row in app_rows if row[0]]
    domains = [
        FacetCount(value=row[0], count=int(row[1] or 0)) for row in domain_rows if row[0]
    ]
    return apps, domains


def fetch_spans(
    db: DatabaseManager,
    event_id: str,
    *,
    matched_span_keys: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    with db.session() as session:
        stmt = select(OCRSpanRecord).where(OCRSpanRecord.capture_id == event_id)
        if matched_span_keys:
            stmt = stmt.where(OCRSpanRecord.span_key.in_(list(matched_span_keys)))
        stmt = stmt.order_by(OCRSpanRecord.start.asc())
        rows = session.execute(stmt).scalars().all()
    return [
        {
            "span_key": row.span_key,
            "span_id": row.span_key,
            "start": row.start,
            "end": row.end,
            "conf": row.confidence,
            "bbox": row.bbox,
            "text": row.text,
        }
        for row in rows
    ]


def _apply_filters(stmt, filters: EventListFilters):
    if filters.start_utc and filters.end_utc:
        stmt = stmt.where(EventRecord.ts_start.between(filters.start_utc, filters.end_utc))
    elif filters.start_utc:
        stmt = stmt.where(EventRecord.ts_start >= filters.start_utc)
    elif filters.end_utc:
        stmt = stmt.where(EventRecord.ts_start <= filters.end_utc)

    if filters.apps:
        stmt = stmt.where(EventRecord.app_name.in_(list(filters.apps)))
    if filters.domains:
        stmt = stmt.where(EventRecord.domain.in_(list(filters.domains)))

    if filters.process:
        pattern = f"%{filters.process.lower()}%"
        stmt = stmt.where(func.lower(EventRecord.app_name).like(pattern))
    if filters.window_title:
        pattern = f"%{filters.window_title.lower()}%"
        stmt = stmt.where(func.lower(EventRecord.window_title).like(pattern))

    if filters.has_screenshot is True:
        stmt = stmt.where(EventRecord.screenshot_path.is_not(None))
    elif filters.has_screenshot is False:
        stmt = stmt.where(EventRecord.screenshot_path.is_(None))

    if filters.has_focus is True:
        stmt = stmt.where(EventRecord.focus_path.is_not(None))
    elif filters.has_focus is False:
        stmt = stmt.where(EventRecord.focus_path.is_(None))

    if filters.event_ids:
        stmt = stmt.where(EventRecord.event_id.in_(list(filters.event_ids)))

    return stmt


def _apply_cursor(stmt, cursor: EventCursor | None):
    if not cursor:
        return stmt
    if cursor.direction == "asc":
        return stmt.where(
            or_(
                EventRecord.ts_start > cursor.ts_start,
                and_(
                    EventRecord.ts_start == cursor.ts_start,
                    EventRecord.event_id > cursor.event_id,
                ),
            )
        )
    return stmt.where(
        or_(
            EventRecord.ts_start < cursor.ts_start,
            and_(
                EventRecord.ts_start == cursor.ts_start,
                EventRecord.event_id < cursor.event_id,
            ),
        )
    )
