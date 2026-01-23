"""Event browsing routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...ux.events_models import EventDetailResponse, EventFacetsResponse, EventListResponse
from ...ux.events_service import EventsService
from ..container import AppContainer


def build_events_router(container: AppContainer) -> APIRouter:
    router = APIRouter()
    service = EventsService(container.config, container.db)

    @router.get("/api/events", response_model=EventListResponse)
    def events_list(
        q: str | None = None,
        start_utc: str | None = None,
        end_utc: str | None = None,
        apps: list[str] | None = Query(None),
        domains: list[str] | None = Query(None),
        process: str | None = None,
        window_title: str | None = None,
        has_screenshot: bool | None = None,
        has_focus: bool | None = None,
        limit: int | None = Query(None, ge=1),
        cursor: str | None = None,
    ) -> EventListResponse:
        try:
            return service.list_events(
                q=q,
                start_utc=start_utc,
                end_utc=end_utc,
                apps=_split_values(apps),
                domains=_split_values(domains),
                process=process,
                window_title=window_title,
                has_screenshot=has_screenshot,
                has_focus=has_focus,
                limit=limit,
                cursor=cursor,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/api/events/facets", response_model=EventFacetsResponse)
    def events_facets(
        q: str | None = None,
        start_utc: str | None = None,
        end_utc: str | None = None,
        apps: list[str] | None = Query(None),
        domains: list[str] | None = Query(None),
        process: str | None = None,
        window_title: str | None = None,
        has_screenshot: bool | None = None,
        has_focus: bool | None = None,
    ) -> EventFacetsResponse:
        try:
            return service.get_facets(
                q=q,
                start_utc=start_utc,
                end_utc=end_utc,
                apps=_split_values(apps),
                domains=_split_values(domains),
                process=process,
                window_title=window_title,
                has_screenshot=has_screenshot,
                has_focus=has_focus,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/api/events/{event_id}", response_model=EventDetailResponse)
    def event_detail(event_id: str) -> EventDetailResponse:
        detail = service.get_event_detail(event_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Event not found")
        return detail

    return router


def _split_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for value in values:
        if value is None:
            continue
        for part in value.split(","):
            stripped = part.strip()
            if stripped:
                items.append(stripped)
    return items
