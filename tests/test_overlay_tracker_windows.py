from __future__ import annotations

import datetime as dt
import sys

import pytest

from autocapture.overlay_tracker.clock import FixedClock

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="windows only")


def test_hotkey_vk_parsing() -> None:
    from autocapture.overlay_tracker.collectors.windows import hotkeys as hk

    assert hk._parse_vk("O") == ord("O")
    assert hk._parse_vk("F12") == hk.VK_F1 + 11
    assert hk._parse_vk("SPACE") == hk.VK_SPACE


def test_hotkey_modifiers_parsing() -> None:
    from autocapture.overlay_tracker.collectors.windows import hotkeys as hk

    value = hk._parse_modifiers(["ctrl", "shift", "alt"])
    assert value & hk.MOD_CONTROL
    assert value & hk.MOD_SHIFT
    assert value & hk.MOD_ALT


def test_input_activity_collector_emits(monkeypatch) -> None:
    from autocapture.overlay_tracker.collectors.windows import input_activity as ia

    clock = FixedClock(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc))
    events = []
    collector = ia.InputActivityCollector(clock=clock, on_event=events.append, poll_ms=1)

    ticks = iter([1, 1, 2, 2, 3])

    def fake_tick():
        try:
            return next(ticks)
        except StopIteration:
            collector._stop.set()
            return None

    monkeypatch.setattr(collector, "_get_last_input_tick", fake_tick)
    monkeypatch.setattr(ia.time, "sleep", lambda *_: None)

    collector.start()
    collector._thread.join(timeout=1.0)

    assert events
    assert all(evt.event_type == "input_activity" for evt in events)
