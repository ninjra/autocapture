from __future__ import annotations

import datetime as dt
from pathlib import Path

from fastapi.testclient import TestClient

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord, OCRSpanRecord, QueryHistoryRecord


def _make_app(tmp_path: Path) -> tuple[TestClient, DatabaseManager]:
    config = AppConfig()
    config.database = DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}")
    config.capture.data_dir = tmp_path
    config.embeddings.model = "local-test"
    db = DatabaseManager(config.database)
    app = create_app(config, db_manager=db)
    return TestClient(app), db


def test_dashboard_redirect(tmp_path: Path) -> None:
    client, _ = _make_app(tmp_path)
    response = client.get("/dashboard", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/"


def test_api_suggest_returns_snippets(tmp_path: Path) -> None:
    client, db = _make_app(tmp_path)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="event-1",
                ts_start=dt.datetime.now(dt.timezone.utc),
                ts_end=None,
                app_name="Notes",
                window_title="Example",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="Hello world from autocapture",
                embedding_vector=None,
                tags={},
            )
        )
        session.add(
            OCRSpanRecord(
                capture_id="event-1",
                span_key="S1",
                start=0,
                end=5,
                text="Hello",
                confidence=0.9,
                bbox={},
            )
        )
        session.add(
            QueryHistoryRecord(
                query_text="hello world",
                normalized_text="hello world",
                count=3,
                last_used_at=dt.datetime.now(dt.timezone.utc),
            )
        )

    response = client.post("/api/suggest", json={"q": "hello"})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload
    assert "snippet" in payload[0]
