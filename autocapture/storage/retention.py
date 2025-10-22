"""Retention manager that enforces image quotas and aging policies."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from ..config import StorageQuotaConfig
from ..logging_utils import get_logger
from .database import DatabaseManager
from .models import CaptureRecord


class RetentionManager:
    def __init__(self, config: StorageQuotaConfig, db: DatabaseManager, media_root: Path) -> None:
        self._config = config
        self._db = db
        self._media_root = media_root
        self._log = get_logger("retention")

    def enforce(self) -> None:
        """Delete oldest captures until storage usage is within quota."""

        usage_gb = self._folder_size_gb(self._media_root)
        if usage_gb < self._config.image_quota_gb:
            return

        delete_count = 0
        with self._db.session() as session:
            stmt = (
                select(CaptureRecord)
                .order_by(CaptureRecord.captured_at.asc())
                .limit(self._config.prune_batch)
            )
            for record in session.scalars(stmt):
                image_path = Path(record.image_path)
                if image_path.exists():
                    image_path.unlink(missing_ok=True)
                session.delete(record)
                delete_count += 1
                if self._folder_size_gb(self._media_root) <= self._config.image_quota_gb:
                    break
        if delete_count:
            self._log.warning("Pruned %s captures to respect quota", delete_count)

    @staticmethod
    def _folder_size_gb(path: Path) -> float:
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024 ** 3)
