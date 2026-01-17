"""Pydantic schemas for agent outputs."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field


class SensitivityInfo(BaseModel):
    contains_pii: bool = False
    contains_secrets: bool = False
    notes: list[str] = Field(default_factory=list)


class ProvenanceInfo(BaseModel):
    model: str
    provider: str
    prompt: str
    created_at_utc: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class TaskItem(BaseModel):
    title: str
    status: Literal["started", "in_progress", "done", "blocked", "unknown"] = "unknown"
    evidence: list[str] = Field(default_factory=list)


class CodeBlock(BaseModel):
    language: str
    text: str


class SqlStatement(BaseModel):
    text: str
    operation: str
    tables: list[str] = Field(default_factory=list)
    parse_error: str | None = None


class EventEnrichmentV2(BaseModel):
    schema_version: Literal["v2"] = "v2"
    event_id: str
    short_summary: str
    what_i_was_doing: str
    apps_and_tools: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    tasks: list[TaskItem] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    importance: float = Field(ge=0.0, le=1.0)
    sensitivity: SensitivityInfo = Field(default_factory=SensitivityInfo)
    keywords: list[str] = Field(default_factory=list)
    code_blocks: list[CodeBlock] = Field(default_factory=list)
    sql_statements: list[SqlStatement] = Field(default_factory=list)
    provenance: ProvenanceInfo


class VisionCaptionV1(BaseModel):
    schema_version: Literal["v1"] = "v1"
    event_id: str
    caption: str
    ui_elements: list[str] = Field(default_factory=list)
    visible_text_summary: str
    sensitivity: SensitivityInfo = Field(default_factory=SensitivityInfo)
    provenance: ProvenanceInfo


class ThreadCitation(BaseModel):
    event_id: str
    ts_start: str
    ts_end: str | None = None


class ThreadSummaryV1(BaseModel):
    schema_version: Literal["v1"] = "v1"
    thread_id: str
    title: str
    summary: str
    key_entities: list[str] = Field(default_factory=list)
    tasks: list[TaskItem] = Field(default_factory=list)
    citations: list[ThreadCitation] = Field(default_factory=list)
    provenance: ProvenanceInfo


class DailyHighlightsV1(BaseModel):
    schema_version: Literal["v1"] = "v1"
    day: str
    summary: str
    highlights: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    open_loops: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    context_switches: list[str] = Field(default_factory=list)
    time_spent_by_app: dict[str, float] = Field(default_factory=dict)
    provenance: ProvenanceInfo
