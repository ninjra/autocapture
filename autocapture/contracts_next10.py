"""Next-10 contract models (versioned, additive)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .contracts import PrivacyFlags


class FrameRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(1)
    frame_id: str
    event_id: Optional[str] = None
    created_at_utc: dt.datetime
    monotonic_ts: float
    monitor_id: Optional[str] = None
    monitor_bounds: Optional[list[int]] = None
    app_name: Optional[str] = None
    window_title: Optional[str] = None
    image_path: Optional[str] = None
    privacy_flags: PrivacyFlags = Field(default_factory=PrivacyFlags)
    frame_hash: Optional[str] = None
    excluded: bool = False
    masked: bool = False


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(1)
    artifact_id: str
    frame_id: str
    artifact_type: str
    engine: Optional[str] = None
    engine_version: Optional[str] = None
    derived_from: dict[str, Any] = Field(default_factory=dict)
    upstream_artifact_ids: list[str] = Field(default_factory=list)
    created_at_utc: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class CitableSpan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(1)
    span_id: str
    artifact_id: str
    frame_id: str
    event_id: Optional[str] = None
    span_hash: str
    text: str
    start_offset: int
    end_offset: int
    bbox: list[int] | None = None
    bbox_norm: list[float] | None = None
    tombstoned: bool = False
    expires_at_utc: dt.datetime | None = None
    legacy_span_key: str | None = None
    created_at_utc: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(1)
    hit_id: str
    query_id: str
    tier: str
    span_id: Optional[str] = None
    event_id: Optional[str] = None
    score: float
    rank: int
    scores_json: dict[str, Any] = Field(default_factory=dict)
    citable: bool = False
    created_at_utc: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class AnswerRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(1)
    answer_id: str
    answer_format_version: int = Field(1)
    query_id: str
    mode: str
    coverage_json: dict[str, Any] = Field(default_factory=dict)
    confidence_json: dict[str, Any] = Field(default_factory=dict)
    budgets_json: dict[str, Any] = Field(default_factory=dict)
    stale: bool = False
    answer_text: str | None = None
    created_at_utc: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class ProvenanceLedgerEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(1)
    entry_id: str
    answer_id: Optional[str] = None
    entry_type: str
    payload_json: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str | None = None
    entry_hash: str
    created_at_utc: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class TierPlanDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(1)
    decision_id: str
    query_id: str
    plan_json: dict[str, Any] = Field(default_factory=dict)
    skipped_json: dict[str, Any] = Field(default_factory=dict)
    reasons_json: dict[str, Any] = Field(default_factory=dict)
    budgets_json: dict[str, Any] = Field(default_factory=dict)
    created_at_utc: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
