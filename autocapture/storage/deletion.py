"""Helpers to delete capture data on demand."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, func, select

from ..fs_utils import safe_unlink
from ..logging_utils import get_logger
from .database import DatabaseManager
from .models import CaptureRecord, EventRecord, SegmentRecord


@dataclass(frozen=True)
class DeleteCounts:
    deleted_captures: int
    deleted_events: int
    deleted_segments: int
    deleted_files: int


def delete_range(
    db: DatabaseManager,
    data_dir: Path,
    *,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    process: str | None = None,
    window_title: str | None = None,
) -> DeleteCounts:
    log = get_logger("storage.delete")
    start = _ensure_aware(start_utc)
    end = _ensure_aware(end_utc)
    if end < start:
        start, end = end, start

    file_paths: set[Path] = set()

    def _collect_path(raw: str | None) -> None:
        if not raw:
            return
        path = Path(raw)
        if not path.is_absolute():
            path = data_dir / path
        file_paths.add(path)

    def _delete(session) -> DeleteCounts:
        captures_stmt = select(CaptureRecord).where(CaptureRecord.captured_at.between(start, end))
        if process:
            captures_stmt = captures_stmt.where(
                func.lower(CaptureRecord.foreground_process) == process.lower()
            )
        if window_title:
            captures_stmt = captures_stmt.where(CaptureRecord.foreground_window == window_title)
        captures = session.execute(captures_stmt).scalars().all()
        capture_ids = [capture.id for capture in captures]
        for capture in captures:
            _collect_path(capture.image_path)

        events_stmt = select(EventRecord).where(EventRecord.ts_start.between(start, end))
        if process:
            events_stmt = events_stmt.where(func.lower(EventRecord.app_name) == process.lower())
        if window_title:
            events_stmt = events_stmt.where(EventRecord.window_title == window_title)
        events = session.execute(events_stmt).scalars().all()
        event_ids = [event.event_id for event in events]
        for event in events:
            _collect_path(event.screenshot_path)

        segments_stmt = select(SegmentRecord).where(SegmentRecord.started_at.between(start, end))
        segments = session.execute(segments_stmt).scalars().all()
        segment_ids = [segment.id for segment in segments]
        for segment in segments:
            _collect_path(segment.video_path)

        deleted_captures = 0
        deleted_events = 0
        deleted_segments = 0
        if capture_ids:
            deleted_captures = (
                session.execute(
                    delete(CaptureRecord).where(CaptureRecord.id.in_(capture_ids))
                ).rowcount
                or 0
            )
        if event_ids:
            deleted_events = (
                session.execute(
                    delete(EventRecord).where(EventRecord.event_id.in_(event_ids))
                ).rowcount
                or 0
            )
        if segment_ids:
            deleted_segments = (
                session.execute(
                    delete(SegmentRecord).where(SegmentRecord.id.in_(segment_ids))
                ).rowcount
                or 0
            )
        return DeleteCounts(
            deleted_captures=deleted_captures,
            deleted_events=deleted_events,
            deleted_segments=deleted_segments,
            deleted_files=0,
        )

    counts = db.transaction(_delete)
    deleted_files = _delete_files(file_paths, log)
    return DeleteCounts(
        deleted_captures=counts.deleted_captures,
        deleted_events=counts.deleted_events,
        deleted_segments=counts.deleted_segments,
        deleted_files=deleted_files,
    )


def _delete_files(paths: set[Path], log) -> int:
    deleted = 0
    for path in paths:
        try:
            if path.exists():
                safe_unlink(path)
                deleted += 1
        except Exception as exc:  # pragma: no cover - filesystem issues
            log.warning("Failed to delete %s: %s", path, exc)
    return deleted


def _ensure_aware(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value
