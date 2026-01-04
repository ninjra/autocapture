"""Retention manager that enforces image quotas and aging policies."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

import datetime as dt

from ..config import RetentionPolicyConfig, StorageQuotaConfig
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
        self._media_root = media_root
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
        with self._db.session() as session:
            stmt = (
                select(EventRecord)
                .where(EventRecord.ts_start < cutoff)
                .where(EventRecord.screenshot_path.is_not(None))
                .order_by(EventRecord.ts_start.asc())
            )
            for event in session.scalars(stmt):
                if event.screenshot_path:
                    path = Path(event.screenshot_path)
                    if path.exists():
                        path.unlink(missing_ok=True)
                event.screenshot_path = None
                removed += 1
        if removed:
            self._log.info("Pruned %s screenshots beyond TTL", removed)
            retention_files_deleted_total.inc(removed)
        return removed

    def _prune_roi_age(self) -> int:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            days=self._retention_config.roi_days
        )
        removed = 0
        with self._db.session() as session:
            stmt = (
                select(CaptureRecord)
                .where(CaptureRecord.captured_at < cutoff)
                .where(CaptureRecord.image_path.is_not(None))
                .order_by(CaptureRecord.captured_at.asc())
            )
            for capture in session.scalars(stmt):
                if capture.image_path:
                    path = Path(capture.image_path)
                    if path.exists():
                        path.unlink(missing_ok=True)
                capture.image_path = None
                removed += 1
        if removed:
            self._log.info("Pruned %s ROI images beyond TTL", removed)
            retention_files_deleted_total.inc(removed)
        return removed

    def _prune_video_age(self) -> int:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            days=self._retention_config.video_days
        )
        removed = 0
        from .models import SegmentRecord

        with self._db.session() as session:
            stmt = (
                select(SegmentRecord)
                .where(SegmentRecord.started_at < cutoff)
                .where(SegmentRecord.video_path.is_not(None))
                .order_by(SegmentRecord.started_at.asc())
            )
            for segment in session.scalars(stmt):
                if segment.video_path:
                    path = Path(segment.video_path)
                    if path.exists():
                        path.unlink(missing_ok=True)
                segment.video_path = None
                removed += 1
        if removed:
            self._log.info("Pruned %s videos beyond TTL", removed)
            retention_files_deleted_total.inc(removed)
        return removed

    def _prune_quota(self) -> int:
        usage_gb = self._folder_size_gb(self._media_root)
        cap_gb = min(self._storage_config.image_quota_gb, self._retention_config.max_media_gb)
        if usage_gb <= cap_gb:
            return 0
        removed = 0
        with self._db.session() as session:
            captures = (
                session.execute(
                    select(CaptureRecord)
                    .where(CaptureRecord.image_path.is_not(None))
                    .order_by(CaptureRecord.captured_at.asc())
                    .limit(self._storage_config.prune_batch)
                )
                .scalars()
                .all()
            )
            for capture in captures:
                if capture.image_path:
                    path = Path(capture.image_path)
                    if path.exists():
                        path.unlink(missing_ok=True)
                capture.image_path = None
                removed += 1
                if self._folder_size_gb(self._media_root) <= cap_gb:
                    break
        if removed:
            self._log.warning("Pruned %s captures to respect quota", removed)
            retention_files_deleted_total.inc(removed)
        return removed

    @staticmethod
    def _folder_size_gb(path: Path) -> float:
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024**3)
