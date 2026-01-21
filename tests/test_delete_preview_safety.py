from __future__ import annotations

import datetime as dt

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord, SegmentRecord
from autocapture.ux.delete_service import DeleteService
from autocapture.ux.models import DeleteCriteria, DeletePreviewRequest


def _make_config(tmp_path):
    return AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
    )


def test_delete_preview_does_not_mutate(tmp_path) -> None:
    config = _make_config(tmp_path)
    db = DatabaseManager(config.database)
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            CaptureRecord(
                captured_at=now,
                foreground_process="test.exe",
                foreground_window="Test Window",
                monitor_id="0",
                privacy_flags={},
            )
        )
        session.add(
            EventRecord(
                ts_start=now,
                app_name="test.exe",
                window_title="Test Window",
                screenshot_hash="hash",
                ocr_text="hello",
            )
        )
        session.add(SegmentRecord(started_at=now))
    service = DeleteService(config, db)
    criteria = DeleteCriteria(
        kind="range",
        start_utc=(now - dt.timedelta(minutes=1)).isoformat(),
        end_utc=(now + dt.timedelta(minutes=1)).isoformat(),
    )
    preview = service.preview(DeletePreviewRequest(criteria=criteria))
    assert preview.counts["captures"] == 1
    assert preview.counts["events"] == 1
    assert preview.counts["segments"] == 1
    with db.session() as session:
        assert session.query(CaptureRecord).count() == 1
        assert session.query(EventRecord).count() == 1
        assert session.query(SegmentRecord).count() == 1
