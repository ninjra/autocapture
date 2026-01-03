from __future__ import annotations

import json

from autocapture.tracking.tracker import HostEventAggregator
from autocapture.tracking.types import (
    ClipboardChangeEvent,
    ForegroundChangeEvent,
    ForegroundContext,
    InputVectorEvent,
)


def _payload(row):
    return json.loads(row.payload_json)


def test_aggregation_flush_and_session() -> None:
    agg = HostEventAggregator(flush_interval_ms=500, idle_grace_ms=300, track_mouse_movement=True)
    ctx = ForegroundContext(process_name="demo.exe", window_title="Demo")

    rows = agg.handle_event(
        ForegroundChangeEvent(ts_ms=1000, new=ctx, old=None), now_ms=1000
    )
    assert rows and rows[0].kind == "foreground_change"

    rows = agg.handle_event(
        InputVectorEvent(ts_ms=1100, device="keyboard", mouse={"events": 1}),
        now_ms=1100,
    )
    assert any(row.kind == "session_start" for row in rows)

    rows = agg.handle_event(
        InputVectorEvent(
            ts_ms=1700,
            device="mouse",
            mouse={"left_clicks": 1, "wheel_events": 1, "wheel_delta": 120, "move_dx": 5, "move_dy": -2},
        ),
        now_ms=1700,
    )
    assert any(row.kind == "input_bucket" for row in rows)
    bucket = next(row for row in rows if row.kind == "input_bucket")
    payload = _payload(bucket)
    assert payload["keyboard_events"] == 1
    assert payload["mouse_left_clicks"] == 1
    assert payload["mouse_wheel_events"] == 1
    assert payload["mouse_wheel_delta"] == 120
    assert payload["mouse_move_dx"] == 5
    assert payload["mouse_move_dy"] == -2

    rows = agg.handle_event(
        ForegroundChangeEvent(
            ts_ms=1500,
            new=ForegroundContext(process_name="other.exe", window_title="Other"),
            old=ctx,
        ),
        now_ms=1500,
    )
    assert any(row.kind == "foreground_change" for row in rows)

    rows = agg.handle_event(
        ClipboardChangeEvent(ts_ms=1600, sequence=3, has_text=True, has_image=False),
        now_ms=1600,
    )
    assert rows and rows[0].kind == "clipboard_change"

    rows = agg.handle_tick(now_ms=2000)
    assert any(row.kind == "session_end" for row in rows)


def test_payload_is_sanitized() -> None:
    agg = HostEventAggregator(flush_interval_ms=100, idle_grace_ms=50, track_mouse_movement=False)
    rows = agg.handle_event(
        InputVectorEvent(ts_ms=1000, device="keyboard", mouse={"events": 2}),
        now_ms=1000,
    )
    rows.extend(agg.handle_tick(now_ms=1200))
    bucket = next(row for row in rows if row.kind == "input_bucket")
    payload = bucket.payload_json
    assert "VKey" not in payload
    assert "MakeCode" not in payload
    assert "scan" not in payload.lower()
