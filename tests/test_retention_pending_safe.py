from __future__ import annotations

import datetime as dt
from pathlib import Path

from autocapture.config import DatabaseConfig, RetentionPolicyConfig, StorageQuotaConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord
from autocapture.storage.retention import RetentionManager


def _write_dummy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")


def test_retention_preserves_pending(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite"
    db = DatabaseManager(DatabaseConfig(url=f"sqlite:///{db_path}"))
    now = dt.datetime.now(dt.timezone.utc)
    pending_path = tmp_path / "media" / "pending.webp"
    done_path = tmp_path / "media" / "done.webp"
    _write_dummy(pending_path)
    _write_dummy(done_path)

    def _seed(session) -> None:
        session.add(
            CaptureRecord(
                id="pending-1",
                captured_at=now - dt.timedelta(days=3),
                image_path=str(pending_path),
                foreground_process="app",
                foreground_window="win",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="pending",
            )
        )
        session.add(
            CaptureRecord(
                id="done-1",
                captured_at=now - dt.timedelta(days=3),
                image_path=str(done_path),
                foreground_process="app",
                foreground_window="win",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )

    db.transaction(_seed)

    retention = RetentionManager(
        StorageQuotaConfig(image_quota_gb=5000, prune_batch=10),
        RetentionPolicyConfig(roi_days=1, protect_recent_minutes=1),
        db,
        tmp_path,
    )
    retention.enforce()

    assert pending_path.exists()
    assert not done_path.exists()
