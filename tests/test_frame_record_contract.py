import datetime as dt

from autocapture.capture.frame_record import build_frame_record_v1, build_privacy_flags, capture_record_kwargs
from autocapture.config import AppConfig, DatabaseConfig, PrivacyConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord
from autocapture.worker.event_worker import EventIngestWorker


def test_frame_record_kwargs_include_required_fields():
    privacy = PrivacyConfig()
    flags = build_privacy_flags(
        privacy,
        excluded=False,
        masked_regions_applied=True,
        offline=True,
    )
    frame = build_frame_record_v1(
        frame_id="frame-1",
        event_id="frame-1",
        captured_at=dt.datetime(2026, 1, 17, 12, 0, tzinfo=dt.timezone.utc),
        monotonic_ts=123.0,
        monitor_id="m1",
        monitor_bounds=(0, 0, 10, 10),
        app_name=None,
        window_title=None,
        image_path=None,
        privacy_flags=flags,
        frame_hash="hash",
    )
    kwargs = capture_record_kwargs(
        frame=frame,
        captured_at=frame.created_at_utc,
        image_path=None,
        focus_path=None,
        foreground_process=None,
        foreground_window=None,
        monitor_id="m1",
        is_fullscreen=False,
        ocr_status="pending",
    )
    required_keys = {
        "id",
        "event_id",
        "created_at_utc",
        "monotonic_ts",
        "monitor_bounds",
        "privacy_flags",
        "frame_hash",
        "schema_version",
    }
    assert required_keys.issubset(kwargs.keys())


def test_privacy_flags_propagate_to_event_tags(tmp_path):
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    db = DatabaseManager(config.database)
    capture_id = "capture-privacy-1"
    with db.session() as session:
        session.add(
            CaptureRecord(
                id=capture_id,
                captured_at=dt.datetime.now(dt.timezone.utc),
                image_path=None,
                foreground_process="App",
                foreground_window="Title",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
                privacy_flags={
                    "excluded": False,
                    "masked_regions_applied": True,
                    "cloud_allowed": False,
                },
            )
        )
        session.add(
            EventRecord(
                event_id=capture_id,
                ts_start=dt.datetime.now(dt.timezone.utc),
                ts_end=None,
                app_name="App",
                window_title="Title",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash=None,
                ocr_text="hello",
                tags={},
            )
        )
    worker = EventIngestWorker(config, db_manager=db, ocr_processor=None)
    with db.session() as session:
        capture = session.get(CaptureRecord, capture_id)
    assert capture is not None
    worker._persist_ocr_results(  # type: ignore[attr-defined]
        capture,
        "hello",
        [],
        event_existing=True,
        screenshot_hash=None,
    )
    with db.session() as session:
        event = session.get(EventRecord, capture_id)
        assert event is not None
        assert event.tags.get("privacy_flags", {}).get("masked_regions_applied") is True
