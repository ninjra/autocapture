"""Pydantic schemas for overlay tracker data flow."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field


class OverlayIdentity(BaseModel):
    identity_type: Literal["title", "url"]
    identity_key: str


class OverlayCollectorContext(BaseModel):
    process_name: str
    window_title: str | None = None
    browser_url: str | None = None
    hwnd: int | None = None


class OverlayCollectorEvent(BaseModel):
    event_type: Literal["foreground", "input_activity"]
    ts_utc: dt.datetime
    context: OverlayCollectorContext
    collector: str
    metadata: dict = Field(default_factory=dict)


class OverlayPersistEvent(BaseModel):
    event_type: Literal["foreground", "input_activity", "hotkey", "state_change"]
    ts_utc: dt.datetime
    process_name: str
    raw_window_title: str | None = None
    raw_browser_url: str | None = None
    identity_type: str | None = None
    identity_key: str | None = None
    collector: str
    schema_version: str = "v1"
    app_version: str | None = None
    payload: dict = Field(default_factory=dict)


class OverlayProjectSummary(BaseModel):
    project_id: int
    name: str


class OverlayItemSummary(BaseModel):
    item_id: int
    project_id: int
    display_name: str | None
    process_name: str
    window_title: str | None
    browser_url: str | None
    last_activity_at_utc: dt.datetime
    state: str
    snooze_until_utc: dt.datetime | None


class OverlayEventEvidence(BaseModel):
    event_id: int
    ts_utc: dt.datetime
    event_type: str
    process_name: str
    raw_window_title: str | None
    raw_browser_url: str | None
    collector: str
    schema_version: str
    app_version: str | None
    payload: dict
