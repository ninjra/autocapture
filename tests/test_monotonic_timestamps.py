import datetime as dt

from autocapture.capture.frame_record import build_frame_record_v1, build_privacy_flags
from autocapture.config import PrivacyConfig
from autocapture.time_utils import elapsed_ms


def test_elapsed_ms_non_negative_with_wall_clock_jump():
    start = 100.0
    end = 90.0
    assert elapsed_ms(start, end) == 0.0


def test_frame_ordering_uses_monotonic_ts():
    privacy = PrivacyConfig()
    flags = build_privacy_flags(
        privacy,
        excluded=False,
        masked_regions_applied=False,
        capture_paused=False,
        offline=True,
    )
    t0 = dt.datetime(2026, 1, 17, 12, 0, tzinfo=dt.timezone.utc)
    frame_a = build_frame_record_v1(
        frame_id="A",
        event_id="A",
        captured_at=t0,
        monotonic_ts=10.0,
        monitor_id="m1",
        monitor_bounds=(0, 0, 10, 10),
        app_name="App",
        window_title="Title",
        image_path=None,
        privacy_flags=flags,
        frame_hash=None,
    )
    frame_b = build_frame_record_v1(
        frame_id="B",
        event_id="B",
        captured_at=t0 - dt.timedelta(minutes=5),
        monotonic_ts=11.0,
        monitor_id="m1",
        monitor_bounds=(0, 0, 10, 10),
        app_name="App",
        window_title="Title",
        image_path=None,
        privacy_flags=flags,
        frame_hash=None,
    )
    ordered = sorted([frame_b, frame_a], key=lambda item: item.monotonic_ts)
    assert [item.frame_id for item in ordered] == ["A", "B"]
