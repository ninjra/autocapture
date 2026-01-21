from __future__ import annotations

from uuid import uuid4

import pytest

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.tracking.store import SqliteHostEventStore, resolve_tracking_db_path, safe_payload
from autocapture.tracking.types import RawInputRow


@pytest.mark.anyio
async def test_tracking_events_requires_unlock_and_returns_rows(
    tmp_path, monkeypatch, async_client_factory
) -> None:
    monkeypatch.setenv("AUTOCAPTURE_TEST_MODE", "0")
    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": True, "raw_event_stream_enabled": True},
        embed={"text_model": "local-test"},
        security={"local_unlock_enabled": True, "provider": "test"},
    )
    db_path = resolve_tracking_db_path(config.tracking, tmp_path)
    store = SqliteHostEventStore(db_path, config=config.tracking)
    store.init_schema()
    store.insert_raw_events(
        [
            RawInputRow(
                id=str(uuid4()),
                ts_ms=1000,
                monotonic_ms=1000,
                device="mouse",
                kind="mouse_move",
                session_id=None,
                app_name="demo.exe",
                window_title="Demo",
                payload_json=safe_payload({"dx": 1, "dy": 2}),
            )
        ]
    )
    store.close()

    app = create_app(config)
    async with async_client_factory(app) as client:
        response = await client.get("/api/tracking/events")
        assert response.status_code == 401

        unlock = await client.post("/api/unlock")
        token = unlock.json().get("token")
        assert token

        response = await client.get(
            "/api/tracking/events",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["events"]
        assert payload["events"][0]["kind"] == "mouse_move"
