from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord, OCRSpanRecord


class BadCitationLLM:
    async def generate_answer(
        self,
        system_prompt: str,
        query: str,
        context_pack_text: str,
        *,
        temperature: float | None = None,
    ) -> str:
        return (
            "```json\n"
            '{"schema_version":2,"claims":[{"text":"Answer","citations":[{"evidence_id":"E999","line_start":1,"line_end":1}]}]}'
            "\n```"
        )


@pytest.mark.anyio
async def test_answer_citations_subset(tmp_path: Path, monkeypatch, async_client_factory) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.capture.data_dir = tmp_path
    config.embed.text_model = "local-test"
    config.model_stages.query_refine.enabled = False
    db = DatabaseManager(config.database)

    with db.session() as session:
        session.add(
            CaptureRecord(
                id="event-1",
                captured_at=dt.datetime.now(dt.timezone.utc),
                image_path=None,
                foreground_process="Docs",
                foreground_window="Notes",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.flush()
        session.add(
            EventRecord(
                event_id="event-1",
                ts_start=dt.datetime.now(dt.timezone.utc),
                ts_end=None,
                app_name="Docs",
                window_title="Notes",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="Meeting notes about roadmap",
                embedding_vector=None,
                tags={},
            )
        )
        session.add(
            OCRSpanRecord(
                capture_id="event-1",
                span_key="S1",
                start=23,
                end=30,
                text="roadmap",
                confidence=0.9,
                bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            )
        )

    def _mock_select(self, stage: str, *, routing_override=None):
        return BadCitationLLM(), type("Decision", (), {"temperature": 0.2, "stage": stage})()

    monkeypatch.setattr("autocapture.model_ops.router.StageRouter.select_llm", _mock_select)

    app = create_app(config, db_manager=db)
    async with async_client_factory(app) as client:
        response = await client.post(
            "/api/answer", json={"query": "roadmap", "extractive_only": False}
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"] == []
    assert payload["mode"] == "BLOCKED"


@pytest.mark.anyio
async def test_answer_json_includes_evidence_payload(tmp_path: Path, async_client_factory) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.capture.data_dir = tmp_path
    config.embed.text_model = "local-test"
    config.model_stages.query_refine.enabled = False
    db = DatabaseManager(config.database)

    with db.session() as session:
        session.add(
            CaptureRecord(
                id="event-json-1",
                captured_at=dt.datetime.now(dt.timezone.utc),
                image_path=None,
                foreground_process="Docs",
                foreground_window="Notes",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.flush()
        session.add(
            EventRecord(
                event_id="event-json-1",
                ts_start=dt.datetime.now(dt.timezone.utc),
                ts_end=None,
                app_name="Docs",
                window_title="Notes",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="Meeting notes about roadmap",
                embedding_vector=None,
                tags={},
            )
        )
        session.add(
            OCRSpanRecord(
                capture_id="event-json-1",
                span_key="S1",
                start=0,
                end=7,
                text="Meeting",
                confidence=0.9,
                bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            )
        )

    app = create_app(config, db_manager=db)
    async with async_client_factory(app) as client:
        response = await client.post(
            "/api/answer",
            json={"query": "roadmap", "extractive_only": True, "output_format": "json"},
        )
    assert response.status_code == 200
    payload = response.json()
    response_json = payload["response_json"]
    assert response_json is not None
    assert response_json["citations"]
    assert response_json["evidence"]
    assert response_json["evidence"][0]["event_id"] == "event-json-1"
    assert response_json["evidence"][0]["ts_start"]


@pytest.mark.anyio
async def test_time_query_timeline_includes_citations(tmp_path: Path, async_client_factory) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.capture.data_dir = tmp_path
    config.embed.text_model = "local-test"
    config.model_stages.query_refine.enabled = False
    db = DatabaseManager(config.database)

    in_range = EventRecord(
        event_id="event-time-1",
        ts_start=dt.datetime(2026, 1, 15, 17, 30, tzinfo=dt.timezone.utc),
        ts_end=None,
        app_name="Calendar",
        window_title="Schedule",
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash",
        ocr_text="Meeting with team",
        embedding_vector=None,
        tags={},
    )
    with db.session() as session:
        session.add(
            CaptureRecord(
                id="event-time-1",
                captured_at=dt.datetime(2026, 1, 15, 17, 30, tzinfo=dt.timezone.utc),
                image_path=None,
                foreground_process="Calendar",
                foreground_window="Schedule",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.flush()
        session.add(in_range)
        session.add(
            OCRSpanRecord(
                capture_id="event-time-1",
                span_key="S1",
                start=0,
                end=7,
                text="Meeting",
                confidence=0.9,
                bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            )
        )

    app = create_app(config, db_manager=db)
    time_range = ["2026-01-15T17:00:00Z", "2026-01-15T18:00:00Z"]
    async with async_client_factory(app) as client:
        response = await client.post(
            "/api/answer",
            json={
                "query": "yesterday 5 pm",
                "time_range": time_range,
                "output_format": "json",
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"]
    assert "event-time-1" in payload["response_json"]["evidence"][0]["event_id"]
