"""Pydantic models for the deterministic memory store."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ArtifactMeta(BaseModel):
    source_uri: str | None = None
    title: str | None = None
    labels: list[str] = Field(default_factory=list)
    content_type: str | None = None


class PolicyDecision(BaseModel):
    action: str
    redacted_text: str | None = None
    redaction_map: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None


class MemoryIngestResult(BaseModel):
    artifact_id: str | None = None
    doc_id: str | None = None
    payload_sha256: str | None = None
    span_ids: list[str] = Field(default_factory=list)
    span_count: int = 0
    excluded: bool = False
    redacted: bool = False
    warnings: list[str] = Field(default_factory=list)


class MemorySpanHit(BaseModel):
    span_id: str
    doc_id: str
    start: int
    end: int
    section_path: str | None = None
    title: str | None = None
    source_uri: str | None = None
    text: str
    span_sha256: str
    text_sha256: str
    score: float


class MemoryQueryResult(BaseModel):
    spans: list[MemorySpanHit] = Field(default_factory=list)
    retrieval_disabled: bool = False
    reason: str | None = None


class MemoryItemRecord(BaseModel):
    item_id: str
    key: str
    value: str
    item_type: str
    status: str
    tags: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    value_sha256: str
    user_asserted: bool = False


class MemoryItemList(BaseModel):
    items: list[MemoryItemRecord] = Field(default_factory=list)


class MemoryPromoteResult(BaseModel):
    item_id: str
    status: str
    deprecated_item_ids: list[str] = Field(default_factory=list)


class MemorySnapshotResult(BaseModel):
    snapshot_id: str
    output_dir: str
    output_sha256: str
    retrieval_disabled: bool
    span_count: int
    item_count: int


class MemoryVerifyResult(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)


class MemoryGcResult(BaseModel):
    removed_snapshots: int
    removed_dirs: int


class HotnessTouchResult(BaseModel):
    item_id: str
    scope: str
    ts_utc: str
    signal: str
    weight: float
    source: str
    applied: bool
    rate_limited: bool = False
    reason: str | None = None


class HotnessPinResult(BaseModel):
    item_id: str
    scope: str
    ts_utc: str
    pin_level: str
    pin_rank: int
    source: str
    applied: bool


class HotnessUnpinResult(BaseModel):
    item_id: str
    scope: str
    ts_utc: str
    source: str
    removed: bool


class HotnessRankEntry(BaseModel):
    item_id: str
    bin: str
    score: float | None = None
    pin_level: str | None = None


class HotnessRankResult(BaseModel):
    scope: str
    as_of_utc: str
    limit: int
    selected: list[HotnessRankEntry] = Field(default_factory=list)
    pinned_over_budget_hard: int = 0
    pinned_over_budget_soft: int = 0
    counts: dict[str, int] = Field(default_factory=dict)


class HotnessGcResult(BaseModel):
    events_deleted: int
    events_remaining: int


class HotnessStateResult(BaseModel):
    item_id: str
    scope: str
    last_ts_utc: str
    h_fast: float
    h_mid: float
    h_warm: float
    h_cool: float
