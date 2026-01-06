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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class EventRecord(Base):
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    ts_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    ts_end: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    app_name: Mapped[str] = mapped_column(String(256))
    window_title: Mapped[str] = mapped_column(String(512))
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    domain: Mapped[str | None] = mapped_column(String(256), nullable=True)
    screenshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    screenshot_hash: Mapped[str] = mapped_column(String(128))
    ocr_text: Mapped[str] = mapped_column(Text)
    embedding_vector: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    embedding_status: Mapped[str] = mapped_column(String(16), default="pending")
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tags: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
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
    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.entity_id", ondelete="CASCADE")
    )
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

    run_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid4())
    )
    ts: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    sources_fetched: Mapped[dict] = mapped_column(JSON, default=dict)
    proposals: Mapped[dict] = mapped_column(JSON, default=dict)
    eval_results: Mapped[dict] = mapped_column(JSON, default=dict)
    pr_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")


class CaptureRecord(Base):
    __tablename__ = "captures"
    __table_args__ = (
        Index("ix_captures_ocr_status", "ocr_status"),
        Index("ix_captures_ocr_status_heartbeat", "ocr_status", "ocr_heartbeat_at"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    captured_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    foreground_process: Mapped[str] = mapped_column(String(256))
    foreground_window: Mapped[str] = mapped_column(String(512))
    monitor_id: Mapped[str] = mapped_column(String(64))
    is_fullscreen: Mapped[bool] = mapped_column(default=False)
    ocr_status: Mapped[str] = mapped_column(String(16), default="pending")
    ocr_started_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ocr_heartbeat_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ocr_attempts: Mapped[int] = mapped_column(Integer, default=0)
    ocr_last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    spans: Mapped[list["OCRSpanRecord"]] = relationship(
        "OCRSpanRecord", back_populates="capture"
    )
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
    capture_id: Mapped[str] = mapped_column(
        ForeignKey("captures.id", ondelete="CASCADE")
    )
    span_key: Mapped[str] = mapped_column(String(64))
    start: Mapped[int] = mapped_column(Integer)
    end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[dict] = mapped_column(JSON)

    capture: Mapped[CaptureRecord] = relationship(
        "CaptureRecord", back_populates="spans"
    )


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
    capture_id: Mapped[str] = mapped_column(
        ForeignKey("captures.id", ondelete="CASCADE")
    )
    vector: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    model: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    processing_started_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    span_key: Mapped[str] = mapped_column(String(64))

    capture: Mapped[CaptureRecord] = relationship(
        "CaptureRecord", back_populates="embeddings"
    )


class SegmentRecord(Base):
    __tablename__ = "segments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    ended_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
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


class HNSWMappingRecord(Base):
    __tablename__ = "hnsw_mapping"
    __table_args__ = (
        UniqueConstraint("event_id", "span_key", name="uq_hnsw_event_span"),
    )

    label: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), index=True)
    span_key: Mapped[str] = mapped_column(String(64))


class ObservationRecord(Base):
    __tablename__ = "observations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    image_path: Mapped[str] = mapped_column(Text)
    segment_id: Mapped[str | None] = mapped_column(
        ForeignKey("segments.id", ondelete="SET NULL")
    )
    cursor_x: Mapped[int] = mapped_column(Integer)
    cursor_y: Mapped[int] = mapped_column(Integer)
    monitor_id: Mapped[str] = mapped_column(String(64))

    segment: Mapped[SegmentRecord | None] = relationship(
        "SegmentRecord", back_populates="observations"
    )
