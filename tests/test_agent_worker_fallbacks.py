from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from autocapture.worker import agent_worker


def _event(event_id: str, title: str, app: str = "App") -> SimpleNamespace:
    return SimpleNamespace(
        event_id=event_id,
        ts_start=dt.datetime.now(dt.timezone.utc),
        ts_end=None,
        window_title=title,
        app_name=app,
        ocr_text=f"{title} notes",
    )


def test_fallback_vision_payload() -> None:
    event = _event("evt-1", "Login Screen")
    result = agent_worker._fallback_vision_payload(
        event,
        response_model="model-x",
        response_provider="provider-y",
        prompt_id="vision_caption:default",
    )
    assert result.event_id == "evt-1"
    assert result.caption
    assert result.provenance.model == "model-x"


def test_fallback_thread_summary_payload() -> None:
    events = [_event("evt-1", "Doc"), _event("evt-2", "Sheet")]
    result = agent_worker._fallback_thread_summary_payload(
        "thread-1",
        "Work Session",
        events,
        response_model="model-x",
        response_provider="provider-y",
        prompt_id="thread_summary:default",
    )
    assert result.thread_id == "thread-1"
    assert result.title
    assert result.summary
    assert result.citations


def test_fallback_daily_highlights_payload() -> None:
    events = [_event("evt-1", "Email"), _event("evt-2", "Browser")]
    aggregates = [
        SimpleNamespace(app_name="Email", metric_name="seconds_active", metric_value=120.0),
        SimpleNamespace(app_name="Browser", metric_name="seconds_active", metric_value=60.0),
    ]
    result = agent_worker._fallback_daily_highlights_payload(
        "2026-01-23",
        events,
        aggregates,
        response_model="model-x",
        response_provider="provider-y",
        prompt_id="daily_highlights:default",
    )
    assert result.day == "2026-01-23"
    assert result.summary
    assert isinstance(result.time_spent_by_app, dict)
