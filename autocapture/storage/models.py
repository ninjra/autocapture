"""SQLAlchemy ORM models for captures and OCR spans."""

from __future__ import annotations

import datetime as dt
from uuid import uuid4

from sqlalchemy import (
    JSON,
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
