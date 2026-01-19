"""Phase 0 contracts and schema models."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PrivacyFlags(BaseModel):
    model_config = ConfigDict(extra="ignore")

    excluded: bool = False
    masked_regions_applied: bool = False
    cloud_allowed: bool = False
    capture_paused: bool = False


class FrameRecordV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field("v1")
    frame_id: str
    event_id: Optional[str] = None
    created_at_utc: dt.datetime
    monotonic_ts: float
    monitor_id: Optional[str] = None
    monitor_bounds: Optional[list[int]] = None
    app_name: Optional[str] = None
    window_title: Optional[str] = None
    active_window: Optional[str] = None
    image_path: Optional[str] = None
    blob_ref: Optional[str] = None
    privacy_flags: PrivacyFlags = Field(default_factory=PrivacyFlags)
    frame_hash: Optional[str] = None
    phash: Optional[str] = None


class OCRSpanV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field("v1")
    span_id: str
    start_offset: int
    end_offset: int
    bbox_px: list[int] = Field(default_factory=list)
    conf: float = Field(0.0, ge=0.0, le=1.0)
    text: str
    engine: str
    frame_id: str
    frame_hash: Optional[str] = None


class RetrievalResultV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_id: str
    frame_id: str
    frame_hash: Optional[str] = None
    snippet: Optional[str] = None
    snippet_offset: Optional[int] = None
    bbox_px: Optional[list[int]] = None
    non_citable: bool = False
    lexical_score: Optional[float] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    late_interaction_score: Optional[float] = None
    rerank_score: Optional[float] = None
    score: float = 0.0
    dedupe_group_id: Optional[str] = None
