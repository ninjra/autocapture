import datetime as dt

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.backfill import BackfillRunner
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord


def test_backfill_resumes_without_duplicates(tmp_path):
    db_path = tmp_path / "test.db"
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{db_path}"))
    db = DatabaseManager(config.database)
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            CaptureRecord(
                id="cap-1",
                captured_at=now - dt.timedelta(minutes=5),
                image_path=None,
                foreground_process="App",
                foreground_window="Window",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.add(
            CaptureRecord(
                id="cap-2",
                captured_at=now,
                image_path=None,
                foreground_process="App",
                foreground_window="Window",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.add(
            EventRecord(
                event_id="cap-1",
                ts_start=now - dt.timedelta(minutes=5),
                ts_end=None,
                app_name="App",
                window_title="Window",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash=None,
                ocr_text="Alpha  123",
                tags={},
            )
        )
        session.add(
            EventRecord(
                event_id="cap-2",
                ts_start=now,
                ts_end=None,
                app_name="App",
                window_title="Window",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash=None,
                ocr_text="Beta  456",
                tags={},
            )
        )
    runner = BackfillRunner(config, db=db)
    first = runner.run(
        tasks=["captures", "events"],
        batch_size=1,
        max_rows=1,
        reset_checkpoints=True,
    )
    second = runner.run(tasks=["captures", "events"], batch_size=1)
    assert first.captures_updated <= 1
    assert first.events_updated <= 1
    assert second.captures_updated + first.captures_updated <= 2
    assert second.events_updated + first.events_updated <= 2
    with db.session() as session:
        capture = session.get(CaptureRecord, "cap-1")
        assert capture is not None
        assert capture.event_id == "cap-1"
        assert capture.created_at_utc is not None
        event = session.get(EventRecord, "cap-1")
        assert event is not None
        assert event.ocr_text_normalized
