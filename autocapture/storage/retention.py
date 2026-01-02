"""Retention manager that enforces image quotas and aging policies."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

import datetime as dt

from ..config import RetentionPolicyConfig, StorageQuotaConfig
from ..logging_utils import get_logger
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
        """Delete oldest captures until storage usage is within quota."""

        usage_gb = self._folder_size_gb(self._media_root)
        if usage_gb < self._storage_config.image_quota_gb:
            return

        delete_count = 0
        with self._db.session() as session:
            stmt = (
                select(CaptureRecord)
                .order_by(CaptureRecord.captured_at.asc())
                .limit(self._storage_config.prune_batch)
            )
            for record in session.scalars(stmt):
                image_path = Path(record.image_path)
                if image_path.exists():
                    image_path.unlink(missing_ok=True)
                session.delete(record)
                delete_count += 1
                if (
                    self._folder_size_gb(self._media_root)
                    <= self._storage_config.image_quota_gb
                ):
                    break
        if delete_count:
            self._log.warning("Pruned %s captures to respect quota", delete_count)

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
        return removed

    @staticmethod
    def _folder_size_gb(path: Path) -> float:
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024**3)
