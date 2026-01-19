"""SQLAlchemy ORM models for captures and OCR spans."""

from __future__ import annotations

import datetime as dt
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates


class Base(DeclarativeBase):
    pass


class EventRecord(Base):
    __tablename__ = "events"

    __table_args__ = (
        Index(
            "idx_events_embedding_status_heartbeat",
            "embedding_status",
            "embedding_heartbeat_at",
        ),
    )

    event_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    ts_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    ts_end: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    app_name: Mapped[str] = mapped_column(String(256))
    window_title: Mapped[str] = mapped_column(String(512))
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    domain: Mapped[str | None] = mapped_column(String(256), nullable=True)
    screenshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    focus_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    screenshot_hash: Mapped[str] = mapped_column(String(128))
    frame_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ocr_text: Mapped[str] = mapped_column(Text)
    ocr_text_normalized: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_vector: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    embedding_status: Mapped[str] = mapped_column(String(16), default="pending")
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_attempts: Mapped[int] = mapped_column(Integer, default=0)
    embedding_last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_started_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    embedding_heartbeat_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    tags: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )

    @validates("screenshot_hash")
    def _normalize_screenshot_hash(self, _key: str, value: str | None) -> str:
        return value or ""


class AgentJobRecord(Base):
    __tablename__ = "agent_jobs"

    __table_args__ = (
        UniqueConstraint("job_key", name="uq_agent_jobs_job_key"),
        Index("ix_agent_jobs_status", "status"),
        Index("ix_agent_jobs_scheduled_for", "scheduled_for"),
        Index("ix_agent_jobs_type", "job_type"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    job_key: Mapped[str] = mapped_column(String(128))
    job_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="pending")
    event_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    day: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    scheduled_for: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    leased_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    leased_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_expires_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class AgentResultRecord(Base):
    __tablename__ = "agent_results"

    __table_args__ = (
        Index("ix_agent_results_type", "job_type"),
        Index("ix_agent_results_event", "event_id"),
        Index("ix_agent_results_day", "day"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    job_type: Mapped[str] = mapped_column(String(64))
    event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    day: Mapped[str | None] = mapped_column(String(10), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(16))
    output_json: Mapped[dict] = mapped_column(JSON)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class EventEnrichmentRecord(Base):
    __tablename__ = "event_enrichments"

    __table_args__ = (UniqueConstraint("event_id", name="uq_event_enrichments_event_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), index=True)
    result_id: Mapped[str] = mapped_column(String(36))
    schema_version: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class ThreadRecord(Base):
    __tablename__ = "threads"

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    ts_end: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    app_name: Mapped[str] = mapped_column(String(256))
    window_title: Mapped[str] = mapped_column(String(512))
    event_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class ThreadEventRecord(Base):
    __tablename__ = "thread_events"

    __table_args__ = (
        UniqueConstraint("thread_id", "event_id", name="uq_thread_events_thread_event"),
        Index("ix_thread_events_event_id", "event_id"),
        Index("ix_thread_events_thread_id", "thread_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.thread_id", ondelete="CASCADE"))
    event_id: Mapped[str] = mapped_column(ForeignKey("events.event_id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer, default=0)


class ThreadSummaryRecord(Base):
    __tablename__ = "thread_summaries"

    thread_id: Mapped[str] = mapped_column(
        ForeignKey("threads.thread_id", ondelete="CASCADE"), primary_key=True
    )
    schema_version: Mapped[str] = mapped_column(String(16))
    data_json: Mapped[dict] = mapped_column(JSON)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class DailyHighlightsRecord(Base):
    __tablename__ = "daily_highlights"

    __table_args__ = (UniqueConstraint("day", name="uq_daily_highlights_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    day: Mapped[str] = mapped_column(String(10))
    schema_version: Mapped[str] = mapped_column(String(16))
    data_json: Mapped[dict] = mapped_column(JSON)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class TokenVaultRecord(Base):
    __tablename__ = "token_vault"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(32))
    value_ciphertext: Mapped[str] = mapped_column(Text)
    value_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    last_seen: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class EntityRecord(Base):
    __tablename__ = "entities"

    entity_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    entity_type: Mapped[str] = mapped_column(String(32))
    canonical_name: Mapped[str] = mapped_column(String(512))
    canonical_token: Mapped[str] = mapped_column(String(64), unique=True)
    parent_entity_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class EntityAliasRecord(Base):
    __tablename__ = "entity_aliases"

    alias_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    entity_id: Mapped[str] = mapped_column(ForeignKey("entities.entity_id", ondelete="CASCADE"))
    alias_text: Mapped[str] = mapped_column(String(512))
    alias_norm: Mapped[str] = mapped_column(String(512))
    alias_type: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class DailyAggregateRecord(Base):
    __tablename__ = "daily_aggregates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    day: Mapped[str] = mapped_column(String(10), index=True)
    app_name: Mapped[str] = mapped_column(String(256))
    domain: Mapped[str | None] = mapped_column(String(256), nullable=True)
    metric_name: Mapped[str] = mapped_column(String(64))
    metric_value: Mapped[float] = mapped_column(Float)
    derived_from: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class PromptLibraryRecord(Base):
    __tablename__ = "prompt_library"

    prompt_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(64))
    raw_template: Mapped[str] = mapped_column(Text)
    derived_template: Mapped[str] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class PromptOpsRunRecord(Base):
    __tablename__ = "prompt_ops_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid4()))
    ts: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    sources_fetched: Mapped[dict] = mapped_column(JSON, default=dict)
    proposals: Mapped[dict] = mapped_column(JSON, default=dict)
    eval_results: Mapped[dict] = mapped_column(JSON, default=dict)
    pr_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")


class RetrievalTraceRecord(Base):
    __tablename__ = "retrieval_traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    query_text: Mapped[str] = mapped_column(Text)
    rewrites_json: Mapped[dict] = mapped_column(JSON, default=dict)
    fused_results_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class BackfillCheckpointRecord(Base):
    __tablename__ = "backfill_checkpoints"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)


class RuntimeStateRecord(Base):
    __tablename__ = "runtime_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    current_mode: Mapped[str] = mapped_column(String(32), default="ACTIVE_INTERACTIVE")
    pause_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    since_ts: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    last_fullscreen_hwnd: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_fullscreen_process: Mapped[str | None] = mapped_column(String(256), nullable=True)
    last_fullscreen_title: Mapped[str | None] = mapped_column(String(512), nullable=True)


class CaptureRecord(Base):
    __tablename__ = "captures"
    __table_args__ = (
        Index("ix_captures_ocr_status", "ocr_status"),
        Index("ix_captures_ocr_status_heartbeat", "ocr_status", "ocr_heartbeat_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    event_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at_utc: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    monotonic_ts: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    focus_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    foreground_process: Mapped[str] = mapped_column(String(256))
    foreground_window: Mapped[str] = mapped_column(String(512))
    monitor_id: Mapped[str] = mapped_column(String(64))
    monitor_bounds: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    is_fullscreen: Mapped[bool] = mapped_column(default=False)
    privacy_flags: Mapped[dict] = mapped_column(JSON, default=dict)
    frame_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(16), default="v1")
    ocr_status: Mapped[str] = mapped_column(String(16), default="pending")
    ocr_started_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ocr_heartbeat_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ocr_attempts: Mapped[int] = mapped_column(Integer, default=0)
    ocr_last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    spans: Mapped[list["OCRSpanRecord"]] = relationship("OCRSpanRecord", back_populates="capture")
    embeddings: Mapped[list["EmbeddingRecord"]] = relationship(
        "EmbeddingRecord", back_populates="capture"
    )


class OCRSpanRecord(Base):
    __tablename__ = "ocr_spans"
    __table_args__ = (
        UniqueConstraint(
            "capture_id",
            "span_key",
            name="uq_ocr_spans_capture_span_key",
        ),
        Index("ix_ocr_spans_capture_id", "capture_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capture_id: Mapped[str] = mapped_column(ForeignKey("captures.id", ondelete="CASCADE"))
    span_key: Mapped[str] = mapped_column(String(64))
    start: Mapped[int] = mapped_column(Integer)
    end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[dict] = mapped_column(JSON)
    engine: Mapped[str | None] = mapped_column(String(64), nullable=True)
    frame_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(16), default="v1")

    capture: Mapped[CaptureRecord] = relationship("CaptureRecord", back_populates="spans")


class EmbeddingRecord(Base):
    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint(
            "capture_id",
            "span_key",
            "model",
            name="uq_embeddings_capture_span_model",
        ),
        ForeignKeyConstraint(
            ["capture_id", "span_key"],
            ["ocr_spans.capture_id", "ocr_spans.span_key"],
            ondelete="CASCADE",
        ),
        Index("ix_embeddings_status", "status"),
        Index("ix_embeddings_status_heartbeat", "status", "heartbeat_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capture_id: Mapped[str] = mapped_column(ForeignKey("captures.id", ondelete="CASCADE"))
    vector: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    model: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    processing_started_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    span_key: Mapped[str] = mapped_column(String(64))
    frame_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)

    capture: Mapped[CaptureRecord] = relationship("CaptureRecord", back_populates="embeddings")


class SegmentRecord(Base):
    __tablename__ = "segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    state: Mapped[str] = mapped_column(String(32), default="recording")
    video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    encoder: Mapped[str | None] = mapped_column(String(64), nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    observations: Mapped[list["ObservationRecord"]] = relationship(
        "ObservationRecord", back_populates="segment"
    )


class QueryHistoryRecord(Base):
    __tablename__ = "query_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(Text, index=True)
    count: Mapped[int] = mapped_column(Integer, default=1)
    last_used_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class ObservationRecord(Base):
    __tablename__ = "observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    image_path: Mapped[str] = mapped_column(Text)
    segment_id: Mapped[str | None] = mapped_column(ForeignKey("segments.id", ondelete="SET NULL"))
    cursor_x: Mapped[int] = mapped_column(Integer)
    cursor_y: Mapped[int] = mapped_column(Integer)
    monitor_id: Mapped[str] = mapped_column(String(64))

    segment: Mapped[SegmentRecord | None] = relationship(
        "SegmentRecord", back_populates="observations"
    )


class OverlayProjectRecord(Base):
    __tablename__ = "overlay_projects"

    __table_args__ = (UniqueConstraint("name", name="uq_overlay_projects_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class OverlayItemRecord(Base):
    __tablename__ = "overlay_items"

    __table_args__ = (
        Index("ix_overlay_items_last_activity", "last_activity_at_utc"),
        Index("ix_overlay_items_project_last_activity", "project_id", "last_activity_at_utc"),
        Index("ix_overlay_items_snooze_until", "snooze_until_utc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("overlay_projects.id", ondelete="CASCADE"),
        index=True,
    )
    display_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    last_process_name: Mapped[str] = mapped_column(String(256))
    last_window_title_raw: Mapped[str | None] = mapped_column(String(512), nullable=True)
    last_browser_url_raw: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    identity_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    identity_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    state: Mapped[str] = mapped_column(String(16), default="idle")
    last_activity_at_utc: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    snooze_until_utc: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )

    project: Mapped[OverlayProjectRecord] = relationship("OverlayProjectRecord")


class OverlayItemIdentityRecord(Base):
    __tablename__ = "overlay_item_identities"

    __table_args__ = (
        UniqueConstraint(
            "identity_type",
            "identity_key",
            name="uq_overlay_item_identities_key",
        ),
        Index("ix_overlay_item_identities_item", "item_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(
        ForeignKey("overlay_items.id", ondelete="CASCADE"),
        index=True,
    )
    identity_type: Mapped[str] = mapped_column(String(16))
    identity_key: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )

    item: Mapped[OverlayItemRecord] = relationship("OverlayItemRecord")


class OverlayEventRecord(Base):
    __tablename__ = "overlay_events"

    __table_args__ = (
        Index("ix_overlay_events_ts", "ts_utc"),
        Index("ix_overlay_events_item_ts", "item_id", "ts_utc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_id: Mapped[int] = mapped_column(
        ForeignKey("overlay_items.id", ondelete="CASCADE"),
        index=True,
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("overlay_projects.id", ondelete="CASCADE"),
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(32))
    ts_utc: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    process_name: Mapped[str] = mapped_column(String(256))
    raw_window_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    raw_browser_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    identity_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    identity_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    collector: Mapped[str] = mapped_column(String(32))
    schema_version: Mapped[str] = mapped_column(String(16), default="v1")
    app_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)

    item: Mapped[OverlayItemRecord] = relationship("OverlayItemRecord")
    project: Mapped[OverlayProjectRecord] = relationship("OverlayProjectRecord")


class OverlayKvRecord(Base):
    __tablename__ = "overlay_kv"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value_json: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class FrameRecord(Base):
    __tablename__ = "frame_records"

    __table_args__ = (
        Index("ix_frame_records_event_id", "event_id"),
        Index("ix_frame_records_captured_at", "captured_at_utc"),
    )

    frame_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_id: Mapped[str | None] = mapped_column(
        ForeignKey("events.event_id", ondelete="SET NULL"),
        nullable=True,
    )
    captured_at_utc: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    monotonic_ts: Mapped[float] = mapped_column(Float)
    monitor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    monitor_bounds: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    app_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    window_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    media_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    privacy_flags: Mapped[dict] = mapped_column(JSON, default=dict)
    frame_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    masked: Mapped[bool] = mapped_column(Boolean, default=False)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class ArtifactRecord(Base):
    __tablename__ = "artifact_records"

    __table_args__ = (Index("ix_artifact_records_frame", "frame_id"),)

    artifact_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    frame_id: Mapped[str] = mapped_column(
        ForeignKey("frame_records.frame_id", ondelete="CASCADE")
    )
    event_id: Mapped[str | None] = mapped_column(
        ForeignKey("events.event_id", ondelete="SET NULL"),
        nullable=True,
    )
    artifact_type: Mapped[str] = mapped_column(String(64))
    engine: Mapped[str | None] = mapped_column(String(64), nullable=True)
    engine_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    derived_from: Mapped[dict] = mapped_column(JSON, default=dict)
    upstream_artifact_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class CitableSpanRecord(Base):
    __tablename__ = "citable_spans"

    __table_args__ = (
        UniqueConstraint("span_hash", name="uq_citable_spans_span_hash"),
        Index("ix_citable_spans_event", "event_id"),
        Index("ix_citable_spans_frame", "frame_id"),
    )

    span_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    artifact_id: Mapped[str] = mapped_column(
        ForeignKey("artifact_records.artifact_id", ondelete="CASCADE")
    )
    frame_id: Mapped[str] = mapped_column(
        ForeignKey("frame_records.frame_id", ondelete="CASCADE")
    )
    event_id: Mapped[str | None] = mapped_column(
        ForeignKey("events.event_id", ondelete="SET NULL"),
        nullable=True,
    )
    span_hash: Mapped[str] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text)
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    bbox_norm: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    tombstoned: Mapped[bool] = mapped_column(Boolean, default=False)
    expires_at_utc: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    legacy_span_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class QueryRecord(Base):
    __tablename__ = "query_records"

    __table_args__ = (Index("ix_query_records_created_at", "created_at"),)

    query_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(Text)
    filters_json: Mapped[dict] = mapped_column(JSON, default=dict)
    query_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    budgets_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class TierPlanDecisionRecord(Base):
    __tablename__ = "tier_plan_decisions"

    __table_args__ = (Index("ix_tier_plan_decisions_query", "query_id"),)

    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_id: Mapped[str] = mapped_column(
        ForeignKey("query_records.query_id", ondelete="CASCADE")
    )
    plan_json: Mapped[dict] = mapped_column(JSON, default=dict)
    skipped_json: Mapped[dict] = mapped_column(JSON, default=dict)
    reasons_json: Mapped[dict] = mapped_column(JSON, default=dict)
    budgets_json: Mapped[dict] = mapped_column(JSON, default=dict)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class RetrievalHitRecord(Base):
    __tablename__ = "retrieval_hits"

    __table_args__ = (
        Index("ix_retrieval_hits_query", "query_id"),
        Index("ix_retrieval_hits_tier_rank", "tier", "rank"),
        Index("ix_retrieval_hits_span", "span_id"),
    )

    hit_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_id: Mapped[str] = mapped_column(
        ForeignKey("query_records.query_id", ondelete="CASCADE")
    )
    tier: Mapped[str] = mapped_column(String(32))
    span_id: Mapped[str | None] = mapped_column(
        ForeignKey("citable_spans.span_id", ondelete="SET NULL"), nullable=True
    )
    event_id: Mapped[str | None] = mapped_column(
        ForeignKey("events.event_id", ondelete="SET NULL"), nullable=True
    )
    score: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer, default=0)
    scores_json: Mapped[dict] = mapped_column(JSON, default=dict)
    citable: Mapped[bool] = mapped_column(Boolean, default=False)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class AnswerRecord(Base):
    __tablename__ = "answer_records"

    __table_args__ = (Index("ix_answer_records_query", "query_id"),)

    answer_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    query_id: Mapped[str] = mapped_column(
        ForeignKey("query_records.query_id", ondelete="CASCADE")
    )
    mode: Mapped[str] = mapped_column(String(32))
    coverage_json: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence_json: Mapped[dict] = mapped_column(JSON, default=dict)
    budgets_json: Mapped[dict] = mapped_column(JSON, default=dict)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    stale: Mapped[bool] = mapped_column(Boolean, default=False)
    answer_format_version: Mapped[int] = mapped_column(Integer, default=1)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class AnswerCitationRecord(Base):
    __tablename__ = "answer_citations"

    __table_args__ = (
        UniqueConstraint("answer_id", "sentence_id", "span_id", name="uq_answer_citations"),
        Index("ix_answer_citations_answer", "answer_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[str] = mapped_column(
        ForeignKey("answer_records.answer_id", ondelete="CASCADE")
    )
    sentence_id: Mapped[str] = mapped_column(String(128))
    sentence_index: Mapped[int] = mapped_column(Integer, default=0)
    span_id: Mapped[str | None] = mapped_column(
        ForeignKey("citable_spans.span_id", ondelete="SET NULL"), nullable=True
    )
    citable: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class ProvenanceLedgerEntryRecord(Base):
    __tablename__ = "provenance_ledger_entries"

    __table_args__ = (
        Index("ix_provenance_ledger_answer", "answer_id"),
        Index("ix_provenance_ledger_hash", "entry_hash"),
    )

    entry_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    answer_id: Mapped[str | None] = mapped_column(
        ForeignKey("answer_records.answer_id", ondelete="CASCADE"),
        nullable=True,
    )
    entry_type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    prev_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    entry_hash: Mapped[str] = mapped_column(String(128))
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class TierStatsRecord(Base):
    __tablename__ = "tier_stats"

    query_class: Mapped[str] = mapped_column(String(64), primary_key=True)
    tier: Mapped[str] = mapped_column(String(32), primary_key=True)
    window_n: Mapped[int] = mapped_column(Integer, default=0)
    help_rate: Mapped[float] = mapped_column(Float, default=0.0)
    p50_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p95_ms: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )


class SchemaMigrationRecord(Base):
    __tablename__ = "schema_migrations"

    version: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    applied_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
