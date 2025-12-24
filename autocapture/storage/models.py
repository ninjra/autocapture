"""SQLAlchemy ORM models for captures and OCR spans."""

from __future__ import annotations

import datetime as dt
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class CaptureRecord(Base):
    __tablename__ = "captures"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    captured_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    image_path: Mapped[str] = mapped_column(Text)
    foreground_process: Mapped[str] = mapped_column(String(256))
    foreground_window: Mapped[str] = mapped_column(String(512))
    monitor_id: Mapped[str] = mapped_column(String(64))
    is_fullscreen: Mapped[bool] = mapped_column(default=False)
    ocr_status: Mapped[str] = mapped_column(String(16), default="pending")

    spans: Mapped[list["OCRSpanRecord"]] = relationship(
        "OCRSpanRecord", back_populates="capture"
    )
    embeddings: Mapped[list["EmbeddingRecord"]] = relationship(
        "EmbeddingRecord", back_populates="capture"
    )


class OCRSpanRecord(Base):
    __tablename__ = "ocr_spans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capture_id: Mapped[str] = mapped_column(
        ForeignKey("captures.id", ondelete="CASCADE")
    )
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[dict] = mapped_column(JSON)

    capture: Mapped[CaptureRecord] = relationship(
        "CaptureRecord", back_populates="spans"
    )


class EmbeddingRecord(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capture_id: Mapped[str] = mapped_column(
        ForeignKey("captures.id", ondelete="CASCADE")
    )
    span_id: Mapped[int | None] = mapped_column(
        ForeignKey("ocr_spans.id", ondelete="SET NULL")
    )
    vector: Mapped[list[float]] = mapped_column(JSON)
    model: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )

    capture: Mapped[CaptureRecord] = relationship(
        "CaptureRecord", back_populates="embeddings"
    )
    span: Mapped[OCRSpanRecord] = relationship("OCRSpanRecord")


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

    observations: Mapped[list["ObservationRecord"]] = relationship(
        "ObservationRecord", back_populates="segment"
    )


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
