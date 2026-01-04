from __future__ import annotations

import datetime as dt
from pathlib import Path

from autocapture.config import AppConfig, DatabaseConfig, RetentionPolicyConfig, StorageQuotaConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, SegmentRecord
from autocapture.storage.retention import RetentionManager


def test_retention_prunes_media(tmp_path: Path) -> None:
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
        retention=RetentionPolicyConfig(roi_days=1, video_days=1, max_media_gb=1),
        storage=StorageQuotaConfig(image_quota_gb=10, prune_grace_days=1, prune_batch=10),
    )
    db = DatabaseManager(config.database)

    roi_path = tmp_path / "media" / "roi" / "old.webp"
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.write_text("roi", encoding="utf-8")

    video_path = tmp_path / "media" / "video" / "old.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_text("video", encoding="utf-8")

    with db.session() as session:
        session.add(
            CaptureRecord(
                id="cap-1",
                captured_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2),
                image_path=str(roi_path),
                foreground_process="test",
                foreground_window="test",
                monitor_id="1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.add(
            SegmentRecord(
                id="seg-1",
                started_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2),
                ended_at=None,
                state="closed",
                video_path=str(video_path),
            )
        )

    retention = RetentionManager(config.storage, config.retention, db, tmp_path)
    retention.enforce()

    with db.session() as session:
        capture = session.get(CaptureRecord, "cap-1")
        segment = session.get(SegmentRecord, "seg-1")

    assert capture is not None
    assert capture.image_path is None
    assert segment is not None
    assert segment.video_path is None
    assert not roi_path.exists()
    assert not video_path.exists()
