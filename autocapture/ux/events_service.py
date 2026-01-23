"""Events browsing service for UX surfaces."""

from __future__ import annotations

import base64
import datetime as dt
import json
from typing import Sequence

from ..config import AppConfig
from ..indexing.lexical_index import LexicalIndex
from ..storage.database import DatabaseManager
from ..storage.queries.events import (
    EventCursor,
    EventListFilters,
    fetch_spans,
    get_event,
    list_events,
    list_facets,
)
from .events_models import (
    EventDetailResponse,
    EventFacetsResponse,
    EventListItem,
    EventListResponse,
    FacetBucket,
)


class EventsService:
    def __init__(self, config: AppConfig, db: DatabaseManager) -> None:
        self._config = config
        self._db = db
        self._lexical = LexicalIndex(db)

    def list_events(
        self,
        *,
        q: str | None = None,
        start_utc: str | None = None,
        end_utc: str | None = None,
        apps: Sequence[str] | None = None,
        domains: Sequence[str] | None = None,
        process: str | None = None,
        window_title: str | None = None,
        has_screenshot: bool | None = None,
        has_focus: bool | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> EventListResponse:
        page_limit = _clamp_limit(limit)
        start_dt = _parse_datetime(start_utc)
        end_dt = _parse_datetime(end_utc)
        event_ids = _resolve_event_ids(self._lexical, q, page_limit)
        if q and not event_ids:
            return EventListResponse(items=[], next_cursor=None)
        cursor_value = _decode_cursor(cursor) if cursor else None
        filters = EventListFilters(
            start_utc=start_dt,
            end_utc=end_dt,
            apps=_normalize_values(apps),
            domains=_normalize_values(domains),
            process=(process or "").strip() or None,
            window_title=(window_title or "").strip() or None,
            has_screenshot=has_screenshot,
            has_focus=has_focus,
            event_ids=event_ids or (),
        )
        events = list_events(self._db, filters=filters, cursor=cursor_value, limit=page_limit)
        next_cursor = None
        if events and len(events) >= page_limit:
            last = events[-1]
            next_cursor = _encode_cursor(
                last.ts_start,
                last.event_id,
                direction=cursor_value.direction if cursor_value else "desc",
            )
        items = [self._to_list_item(event, q=q) for event in events]
        return EventListResponse(items=items, next_cursor=next_cursor)

    def get_facets(
        self,
        *,
        q: str | None = None,
        start_utc: str | None = None,
        end_utc: str | None = None,
        apps: Sequence[str] | None = None,
        domains: Sequence[str] | None = None,
        process: str | None = None,
        window_title: str | None = None,
        has_screenshot: bool | None = None,
        has_focus: bool | None = None,
    ) -> EventFacetsResponse:
        start_dt = _parse_datetime(start_utc)
        end_dt = _parse_datetime(end_utc)
        event_ids = _resolve_event_ids(self._lexical, q, 200)
        if q and not event_ids:
            return EventFacetsResponse(apps=[], domains=[])
        filters = EventListFilters(
            start_utc=start_dt,
            end_utc=end_dt,
            apps=_normalize_values(apps),
            domains=_normalize_values(domains),
            process=(process or "").strip() or None,
            window_title=(window_title or "").strip() or None,
            has_screenshot=has_screenshot,
            has_focus=has_focus,
            event_ids=event_ids or (),
        )
        apps_rows, domain_rows = list_facets(self._db, filters=filters)
        return EventFacetsResponse(
            apps=[FacetBucket(value=row.value, count=row.count) for row in apps_rows],
            domains=[FacetBucket(value=row.value, count=row.count) for row in domain_rows],
        )

    def get_event_detail(self, event_id: str) -> EventDetailResponse | None:
        event = get_event(self._db, event_id)
        if not event:
            return None
        spans = fetch_spans(self._db, event.event_id)
        return EventDetailResponse(
            event_id=event.event_id,
            ts_start=event.ts_start,
            ts_end=event.ts_end,
            app_name=event.app_name,
            window_title=event.window_title,
            url=event.url,
            domain=event.domain,
            screenshot_path=event.screenshot_path,
            focus_path=event.focus_path,
            screenshot_hash=event.screenshot_hash,
            ocr_text=event.ocr_text or "",
            ocr_spans=spans,
            tags=event.tags or {},
        )

    def _to_list_item(self, event, *, q: str | None = None) -> EventListItem:
        has_screenshot = bool(event.screenshot_path)
        has_focus = bool(event.focus_path)
        snippet = _make_snippet(event.ocr_text or "", q=q)
        return EventListItem(
            event_id=event.event_id,
            ts_start=event.ts_start,
            ts_end=event.ts_end,
            app_name=event.app_name,
            window_title=event.window_title,
            url=event.url,
            domain=event.domain,
            has_screenshot=has_screenshot,
            has_focus=has_focus,
            screenshot_url=_build_media_url("screenshot", event.event_id, has_screenshot),
            focus_url=_build_media_url("focus", event.event_id, has_focus),
            ocr_snippet=snippet,
        )


def _build_media_url(kind: str, event_id: str, enabled: bool) -> str | None:
    if not enabled:
        return None
    if kind == "focus":
        return f"/api/focus/{event_id}?variant=thumb"
    return f"/api/screenshot/{event_id}?variant=thumb"


def _normalize_values(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    normalized: list[str] = []
    for value in values:
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            normalized.append(stripped)
    return tuple(normalized)


def _parse_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError("Invalid datetime format") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _clamp_limit(limit: int | None) -> int:
    if limit is None:
        return 100
    limit = int(limit)
    if limit < 1:
        return 1
    if limit > 500:
        return 500
    return limit


def _resolve_event_ids(
    lexical: LexicalIndex,
    query: str | None,
    limit: int,
) -> tuple[str, ...] | None:
    query = (query or "").strip()
    if not query:
        return None
    search_limit = max(min(limit * 5, 2000), 200)
    hits = lexical.search(query, limit=search_limit)
    return tuple(hit.event_id for hit in hits if hit.event_id)


def _make_snippet(text: str, *, q: str | None = None, max_len: int = 180) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    if len(cleaned) <= max_len:
        return cleaned
    needle = (q or "").strip().lower()
    if needle:
        idx = cleaned.lower().find(needle)
        if idx >= 0:
            start = max(idx - 40, 0)
            end = min(idx + max_len, len(cleaned))
            snippet = cleaned[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(cleaned):
                snippet = snippet + "..."
            return snippet
    return cleaned[: max_len - 3] + "..."


def _encode_cursor(ts_start: dt.datetime, event_id: str, *, direction: str) -> str:
    if ts_start.tzinfo is None:
        ts_start = ts_start.replace(tzinfo=dt.timezone.utc)
    payload = {
        "ts_start": ts_start.isoformat(),
        "event_id": event_id,
        "direction": direction,
    }
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return base64.urlsafe_b64encode(body).rstrip(b"=").decode("ascii")


def _decode_cursor(cursor: str) -> EventCursor:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid cursor") from exc
    ts_raw = payload.get("ts_start")
    event_id = payload.get("event_id")
    direction = payload.get("direction", "desc")
    if direction not in {"asc", "desc"}:
        raise ValueError("Invalid cursor direction")
    ts = _parse_datetime(ts_raw)
    if not ts or not event_id:
        raise ValueError("Invalid cursor payload")
    return EventCursor(ts_start=ts, event_id=str(event_id), direction=direction)
