"""Retention manager that enforces image quotas and aging policies."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

import datetime as dt
import os

from ..config import RetentionPolicyConfig, StorageQuotaConfig
from ..fs_utils import safe_unlink
from ..logging_utils import get_logger
from ..observability.metrics import retention_files_deleted_total
from .database import DatabaseManager
from .models import CaptureRecord, EventRecord


class RetentionManager:
    def __init__(
        self,
        storage_config: StorageQuotaConfig,
        retention_config: RetentionPolicyConfig,
        db: DatabaseManager,
        media_root: Path,
    ) -> None:
        self._storage_config = storage_config
        self._retention_config = retention_config
        self._db = db
        # Callers pass the data dir; quota checks should apply to the media subtree.
        self._media_root = (
            media_root if media_root.name == "media" else media_root / "media"
        )
        self._log = get_logger("retention")

    def enforce(self) -> None:
        """Apply retention policies for ROI/video media and storage caps."""

        self._prune_roi_age()
        self._prune_video_age()
        self._prune_quota()

    def enforce_screenshot_ttl(self) -> int:
        """Remove screenshots older than TTL but preserve event metadata."""

        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            days=self._retention_config.screenshot_ttl_days
        )
        removed = 0

        paths: list[Path] = []

        def _prune(session) -> None:
            nonlocal removed
            stmt = (
                select(EventRecord)
                .where(EventRecord.ts_start < cutoff)
                .where(EventRecord.screenshot_path.is_not(None))
                .order_by(EventRecord.ts_start.asc())
            )
            for event in session.scalars(stmt):
                if event.screenshot_path:
                    paths.append(Path(event.screenshot_path))
                event.screenshot_path = None
                removed += 1

        self._db.transaction(_prune)
        self._delete_files(paths)
        if removed:
            self._log.info("Pruned {} screenshots beyond TTL", removed)
            retention_files_deleted_total.inc(removed)
        return removed

    def _prune_roi_age(self) -> int:
        safe_statuses = {"done", "failed", "skipped"}
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            days=self._retention_config.roi_days
        )
        protect_cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            minutes=self._retention_config.protect_recent_minutes
        )
        removed = 0

        paths: list[Path] = []

        def _prune(session) -> None:
            nonlocal removed
            stmt = (
                select(CaptureRecord)
                .where(CaptureRecord.captured_at < cutoff)
                .where(CaptureRecord.captured_at < protect_cutoff)
                .where(CaptureRecord.ocr_status.in_(safe_statuses))
                .where(CaptureRecord.image_path.is_not(None))
                .order_by(CaptureRecord.captured_at.asc())
            )
            for capture in session.scalars(stmt):
                if capture.image_path:
                    paths.append(Path(capture.image_path))
                    session.query(EventRecord).filter(
                        EventRecord.screenshot_path == capture.image_path
                    ).update({EventRecord.screenshot_path: None})
                capture.image_path = None
                removed += 1

        self._db.transaction(_prune)
        self._delete_files(paths)
        if removed:
            self._log.info("Pruned {} ROI images beyond TTL", removed)
            retention_files_deleted_total.inc(removed)
        return removed

    def _prune_video_age(self) -> int:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            days=self._retention_config.video_days
        )
        removed = 0
        from .models import SegmentRecord

        paths: list[Path] = []

        def _prune(session) -> None:
            nonlocal removed
            stmt = (
                select(SegmentRecord)
                .where(SegmentRecord.started_at < cutoff)
                .where(SegmentRecord.ended_at.is_not(None))
                .where(SegmentRecord.state != "recording")
                .where(SegmentRecord.video_path.is_not(None))
                .order_by(SegmentRecord.started_at.asc())
            )
            for segment in session.scalars(stmt):
                if segment.video_path:
                    paths.append(Path(segment.video_path))
                segment.video_path = None
                removed += 1

        self._db.transaction(_prune)
        self._delete_files(paths)
        if removed:
            self._log.info("Pruned {} videos beyond TTL", removed)
            retention_files_deleted_total.inc(removed)
        return removed

    def _prune_quota(self) -> int:
        cap_gb = min(
            self._storage_config.image_quota_gb, self._retention_config.max_media_gb
        )
        cap_bytes = int(cap_gb * (1024**3))
        remaining_bytes = self._folder_size_bytes(self._media_root)
        if remaining_bytes <= cap_bytes:
            return 0
        removed = 0
        safe_statuses = {"done", "failed", "skipped"}
        protect_cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            minutes=self._retention_config.protect_recent_minutes
        )

        paths: list[Path] = []

        def _prune(session) -> None:
            nonlocal removed, remaining_bytes
            captures = (
                session.execute(
                    select(CaptureRecord)
                    .where(CaptureRecord.image_path.is_not(None))
                    .where(CaptureRecord.ocr_status.in_(safe_statuses))
                    .where(CaptureRecord.captured_at < protect_cutoff)
                    .order_by(CaptureRecord.captured_at.asc())
                    .limit(self._storage_config.prune_batch)
                )
                .scalars()
                .all()
            )
            for capture in captures:
                if capture.image_path:
                    path = Path(capture.image_path)
                    paths.append(path)
                    session.query(EventRecord).filter(
                        EventRecord.screenshot_path == capture.image_path
                    ).update({EventRecord.screenshot_path: None})
                    try:
                        remaining_bytes -= path.stat().st_size
                    except OSError:
                        pass
                capture.image_path = None
                removed += 1
                if remaining_bytes <= cap_bytes:
                    break

        self._db.transaction(_prune)
        self._delete_files(paths)
        if removed:
            self._log.warning("Pruned {} captures to respect quota", removed)
            retention_files_deleted_total.inc(removed)
        return removed

    def _delete_files(self, paths: list[Path]) -> None:
        for path in paths:
            try:
                if path.exists():
                    safe_unlink(path)
            except Exception as exc:  # pragma: no cover - filesystem issues
                self._log.error("Failed to delete {}: {}", path, exc)

    @staticmethod
    def _folder_size_gb(path: Path) -> float:
        return RetentionManager._folder_size_bytes(path) / (1024**3)

    @staticmethod
    def _folder_size_bytes(path: Path) -> int:
        """Best-effort recursive folder size (bytes) using os.scandir."""

        def _walk(p: Path) -> int:
            total = 0
            try:
                with os.scandir(p) as it:
                    for entry in it:
                        try:
                            if entry.is_file(follow_symlinks=False):
                                total += entry.stat(follow_symlinks=False).st_size
                            elif entry.is_dir(follow_symlinks=False):
                                total += _walk(Path(entry.path))
                        except (FileNotFoundError, PermissionError, OSError):
                            continue
            except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
                return 0
            return total

        return _walk(path)
