from __future__ import annotations

import datetime as dt

from autocapture.config import DatabaseConfig
from autocapture.overlay_tracker.clock import SystemClock
from autocapture.overlay_tracker.store import OverlayTrackerStore
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, RuntimeStateRecord


def test_overlay_health_snapshot_reports_last_capture() -> None:
    db = DatabaseManager(DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    store = OverlayTrackerStore(db, SystemClock())
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            CaptureRecord(
                id="cap-1",
                event_id="evt-1",
                captured_at=now,
                foreground_process="App",
                foreground_window="Window",
                monitor_id="m1",
                privacy_flags={},
            )
        )
        session.add(
            RuntimeStateRecord(
                id=1,
                current_mode="ACTIVE_INTERACTIVE",
                pause_reason=None,
                since_ts=now,
            )
        )
    health = store.capture_health()
    assert health["db_ok"] is True
    assert health["last_capture_at"]
    assert health["runtime_mode"] == "ACTIVE_INTERACTIVE"
