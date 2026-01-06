from __future__ import annotations

import datetime as dt
from pathlib import Path

from autocapture.config import (
    AppConfig,
    DatabaseConfig,
    RetentionPolicyConfig,
    StorageQuotaConfig,
)
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord, SegmentRecord
from autocapture.storage import retention as retention_module
from autocapture.storage.retention import RetentionManager


def test_retention_prunes_media(tmp_path: Path) -> None:
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
        retention=RetentionPolicyConfig(roi_days=1, video_days=1, max_media_gb=1),
        storage=StorageQuotaConfig(
            image_quota_gb=10, prune_grace_days=1, prune_batch=10
        ),
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
                ended_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2),
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


def test_retention_clears_event_screenshot_path(tmp_path: Path) -> None:
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
        retention=RetentionPolicyConfig(roi_days=1, video_days=1, max_media_gb=1),
        storage=StorageQuotaConfig(
            image_quota_gb=10, prune_grace_days=1, prune_batch=10
        ),
    )
    db = DatabaseManager(config.database)
    roi_path = tmp_path / "media" / "roi" / "old.webp"
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.write_text("roi", encoding="utf-8")

    with db.session() as session:
        session.add(
            CaptureRecord(
                id="cap-2",
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
            EventRecord(
                event_id="cap-2",
                ts_start=dt.datetime.now(dt.timezone.utc),
                ts_end=None,
                app_name="test",
                window_title="test",
                url=None,
                domain=None,
                screenshot_path=str(roi_path),
                screenshot_hash="hash",
                ocr_text="text",
                embedding_vector=None,
                embedding_status="pending",
                embedding_model=None,
                tags={},
            )
        )

    retention = RetentionManager(config.storage, config.retention, db, tmp_path)
    retention.enforce()

    with db.session() as session:
        capture = session.get(CaptureRecord, "cap-2")
        event = session.get(EventRecord, "cap-2")

    assert capture is not None
    assert event is not None
    assert capture.image_path is None
    assert event.screenshot_path is None


def test_retention_idempotent(tmp_path: Path) -> None:
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
        retention=RetentionPolicyConfig(roi_days=1, video_days=1, max_media_gb=1),
        storage=StorageQuotaConfig(
            image_quota_gb=10, prune_grace_days=1, prune_batch=10
        ),
    )
    db = DatabaseManager(config.database)
    retention = RetentionManager(config.storage, config.retention, db, tmp_path)
    retention.enforce()
    retention.enforce()


def test_retention_updates_db_before_delete(tmp_path: Path, monkeypatch) -> None:
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
        retention=RetentionPolicyConfig(roi_days=1, video_days=1, max_media_gb=1),
        storage=StorageQuotaConfig(
            image_quota_gb=10, prune_grace_days=1, prune_batch=10
        ),
    )
    db = DatabaseManager(config.database)
    roi_path = tmp_path / "media" / "roi" / "old.webp"
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.write_text("roi", encoding="utf-8")

    with db.session() as session:
        session.add(
            CaptureRecord(
                id="cap-3",
                captured_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2),
                image_path=str(roi_path),
                foreground_process="test",
                foreground_window="test",
                monitor_id="1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )

    def _fail_unlink(path: Path) -> None:
        raise OSError("boom")

    monkeypatch.setattr(retention_module, "safe_unlink", _fail_unlink)

    retention = RetentionManager(config.storage, config.retention, db, tmp_path)
    retention.enforce()

    with db.session() as session:
        capture = session.get(CaptureRecord, "cap-3")

    assert capture is not None
    assert capture.image_path is None
