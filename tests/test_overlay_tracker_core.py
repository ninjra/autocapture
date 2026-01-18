from __future__ import annotations

import datetime as dt
import math

from autocapture.config import DatabaseConfig, OverlayTrackerConfig
from autocapture.overlay_tracker.clock import FixedClock
from autocapture.overlay_tracker.core import hotness, is_stale, normalize_title, resolve_identity
from autocapture.overlay_tracker.schemas import OverlayPersistEvent
from autocapture.overlay_tracker.store import OverlayTrackerStore
from autocapture.storage.database import DatabaseManager


def test_hotness_half_life() -> None:
    now = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    last = now - dt.timedelta(minutes=30)
    value = hotness(last, now, half_life_s=1800)
    assert math.isclose(value, 0.5, rel_tol=1e-6)


def test_hotness_monotonic() -> None:
    now = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    recent = now - dt.timedelta(minutes=5)
    older = now - dt.timedelta(minutes=25)
    assert hotness(recent, now, half_life_s=1800) > hotness(older, now, half_life_s=1800)


def test_stale_boundary() -> None:
    now = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    threshold = now - dt.timedelta(hours=48)
    assert not is_stale(threshold, now, stale_after_s=48 * 3600)
    assert is_stale(threshold - dt.timedelta(seconds=1), now, stale_after_s=48 * 3600)


def test_normalize_title() -> None:
    raw = "  Hello\tWorld  "
    assert normalize_title(raw, max_len=512) == "Hello World"
    assert normalize_title(raw, max_len=5) == "Hello"


def test_identity_resolution_url_enabled() -> None:
    config = OverlayTrackerConfig()
    config.url_plugin.enabled = True
    config.url_plugin.allow_browsers = ["chrome.exe"]
    identity = resolve_identity(
        config,
        process_name="chrome.exe",
        window_title="GitHub",
        browser_url="https://github.com/openai/gpt-4",
    )
    assert identity.identity_type == "url"
    assert identity.identity_key == "github.com/openai/gpt-4"


def test_identity_resolution_fail_closed() -> None:
    config = OverlayTrackerConfig()
    config.url_plugin.enabled = True
    config.url_plugin.allow_browsers = ["chrome.exe"]
    identity = resolve_identity(
        config,
        process_name="chrome.exe",
        window_title="Untitled",
        browser_url="not a url",
    )
    assert identity.identity_type == "title"


def test_default_project_insert_and_retention() -> None:
    clock = FixedClock(dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc))
    db = DatabaseManager(DatabaseConfig(url="sqlite:///:memory:"))
    store = OverlayTrackerStore(db, clock)
    events = [
        OverlayPersistEvent(
            event_type="foreground",
            ts_utc=clock.now() - dt.timedelta(days=10),
            process_name="explorer.exe",
            raw_window_title="Explorer",
            raw_browser_url=None,
            identity_type="title",
            identity_key="explorer.exe:Explorer",
            collector="test",
        ),
        OverlayPersistEvent(
            event_type="foreground",
            ts_utc=clock.now(),
            process_name="notepad.exe",
            raw_window_title="Notes",
            raw_browser_url=None,
            identity_type="title",
            identity_key="notepad.exe:Notes",
            collector="test",
        ),
    ]
    store.record_events(events)
    projects = store.query_projects()
    assert projects
    store.retention_cleanup(event_days=1, event_cap=1, now_utc=clock.now())
    active, stale = store.query_items(clock.now(), stale_after_s=48 * 3600)
    total_evidence = 0
    for item in [*active, *stale]:
        total_evidence += len(store.query_evidence(item.item_id, limit=10))
    assert total_evidence <= 1
