"""Pydantic models for event browsing surfaces."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    ts_start: dt.datetime
    ts_end: dt.datetime | None = None
    app_name: str
    window_title: str
    url: str | None = None
    domain: str | None = None
    has_screenshot: bool = False
    has_focus: bool = False
    screenshot_url: str | None = None
    focus_url: str | None = None
    ocr_snippet: str | None = None


class EventListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[EventListItem] = Field(default_factory=list)
    next_cursor: str | None = None


class FacetBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    count: int


class EventFacetsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apps: list[FacetBucket] = Field(default_factory=list)
    domains: list[FacetBucket] = Field(default_factory=list)


class EventDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    ts_start: dt.datetime
    ts_end: dt.datetime | None = None
    app_name: str
    window_title: str
    url: str | None = None
    domain: str | None = None
    screenshot_path: str | None = None
    focus_path: str | None = None
    screenshot_hash: str
    ocr_text: str
    ocr_spans: list[dict[str, Any]]
    tags: dict[str, Any]
