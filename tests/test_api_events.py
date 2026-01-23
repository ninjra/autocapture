from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord, OCRSpanRecord


def _make_app(tmp_path: Path) -> tuple[object, DatabaseManager]:
    config = AppConfig()
    config.database = DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}")
    config.capture.data_dir = tmp_path
    config.embed.text_model = "local-test"
    db = DatabaseManager(config.database)
    app = create_app(config, db_manager=db)
    return app, db


@pytest.mark.anyio
async def test_api_events_list_paginates(tmp_path: Path, async_client_factory) -> None:
    app, db = _make_app(tmp_path)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    with db.session() as session:
        session.add(
            CaptureRecord(
                id="event-1",
                captured_at=now,
                image_path=None,
                foreground_process="Notes",
                foreground_window="Example",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.flush()
        session.add(
            EventRecord(
                event_id="event-1",
                ts_start=now - dt.timedelta(minutes=2),
                ts_end=None,
                app_name="Notes",
                window_title="Example",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash-1",
                ocr_text="first event",
                embedding_vector=None,
                tags={},
            )
        )
        session.add(
            EventRecord(
                event_id="event-2",
                ts_start=now - dt.timedelta(minutes=1),
                ts_end=None,
                app_name="Browser",
                window_title="Example",
                url="https://example.com",
                domain="example.com",
                screenshot_path=None,
                screenshot_hash="hash-2",
                ocr_text="second event",
                embedding_vector=None,
                tags={},
            )
        )

    async with async_client_factory(app) as client:
        response = await client.get("/api/events?limit=1")
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["items"]) == 1
        assert payload["items"][0]["event_id"] == "event-2"
        assert payload["next_cursor"]

        cursor = payload["next_cursor"]
        response = await client.get(f"/api/events?limit=1&cursor={cursor}")
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["items"]) == 1
        assert payload["items"][0]["event_id"] == "event-1"


@pytest.mark.anyio
async def test_api_events_detail_includes_spans(tmp_path: Path, async_client_factory) -> None:
    app, db = _make_app(tmp_path)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    with db.session() as session:
        session.add(
            CaptureRecord(
                id="event-1",
                captured_at=now,
                image_path=None,
                foreground_process="Notes",
                foreground_window="Example",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.flush()
        session.add(
            EventRecord(
                event_id="event-1",
                ts_start=now,
                ts_end=None,
                app_name="Notes",
                window_title="Example",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash-1",
                ocr_text="hello world",
                embedding_vector=None,
                tags={},
            )
        )
        session.add(
            OCRSpanRecord(
                capture_id="event-1",
                span_key="s1",
                start=0,
                end=5,
                text="hello",
                confidence=0.9,
                bbox={},
            )
        )

    async with async_client_factory(app) as client:
        response = await client.get("/api/events/event-1")
        assert response.status_code == 200
        payload = response.json()
        assert payload["event_id"] == "event-1"
        assert payload["ocr_spans"]
        assert payload["ocr_spans"][0]["span_key"] == "s1"

        legacy = await client.get("/api/event/event-1")
        assert legacy.status_code == 200
        legacy_payload = legacy.json()
        assert legacy_payload["event_id"] == "event-1"
