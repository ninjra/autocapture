from __future__ import annotations

import datetime as dt

from dateutil import tz

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.memory.retrieval import RetrievalService
from autocapture.memory.time_intent import (
    is_time_only_expression,
    parse_time_expression,
    resolve_time_range_for_query,
)
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord


def test_parse_time_expression_last_hour() -> None:
    now = dt.datetime(2026, 1, 16, 18, 0, tzinfo=dt.timezone.utc)
    result = parse_time_expression("an hour ago", now=now, tzinfo=dt.timezone.utc)
    assert result == (now - dt.timedelta(hours=1), now)


def test_parse_time_expression_yesterday_5pm() -> None:
    now = dt.datetime(2026, 1, 16, 18, 0, tzinfo=dt.timezone.utc)
    result = parse_time_expression("yesterday 5 pm", now=now, tzinfo=dt.timezone.utc)
    assert result == (
        dt.datetime(2026, 1, 15, 17, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 1, 15, 18, 0, tzinfo=dt.timezone.utc),
    )


def test_parse_time_expression_at_1700_yesterday() -> None:
    now = dt.datetime(2026, 1, 16, 18, 0, tzinfo=dt.timezone.utc)
    result = parse_time_expression("at 17:00 yesterday", now=now, tzinfo=dt.timezone.utc)
    assert result == (
        dt.datetime(2026, 1, 15, 17, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 1, 15, 18, 0, tzinfo=dt.timezone.utc),
    )


def test_time_only_retrieval_returns_range(tmp_path) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.embed.text_model = "local-test"
    config.qdrant.enabled = False
    db = DatabaseManager(config.database)

    inside = EventRecord(
        event_id="E1",
        ts_start=dt.datetime(2026, 1, 15, 17, 30, tzinfo=dt.timezone.utc),
        ts_end=None,
        app_name="App",
        window_title="Win",
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash1",
        ocr_text="text",
        embedding_vector=None,
        tags={},
    )
    outside = EventRecord(
        event_id="E2",
        ts_start=dt.datetime(2026, 1, 15, 20, 0, tzinfo=dt.timezone.utc),
        ts_end=None,
        app_name="App",
        window_title="Win",
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash2",
        ocr_text="text",
        embedding_vector=None,
        tags={},
    )
    with db.session() as session:
        session.add(inside)
        session.add(outside)

    retrieval = RetrievalService(db, config)
    time_range = (
        dt.datetime(2026, 1, 15, 17, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 1, 15, 18, 0, tzinfo=dt.timezone.utc),
    )
    batch = retrieval.retrieve("", time_range, None, limit=10)
    assert [item.event.event_id for item in batch.results] == ["E1"]


def test_resolve_time_range_for_query_uses_timezone_override() -> None:
    now = dt.datetime(2026, 1, 16, 18, 0, tzinfo=dt.timezone.utc)
    tzinfo = tz.gettz("America/Denver")
    resolved = resolve_time_range_for_query(
        query="yesterday 5 pm",
        time_range=None,
        now=now,
        tzinfo=tzinfo,
    )
    assert resolved is not None


def test_is_time_only_expression() -> None:
    assert is_time_only_expression("yesterday 5 pm") is True
    assert is_time_only_expression("an hour ago") is True
    assert is_time_only_expression("yesterday 5 pm notes") is False
