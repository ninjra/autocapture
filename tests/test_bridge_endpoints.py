from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord


def _sample_png_bytes() -> bytes:
    image = Image.new("RGB", (4, 4), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_config(tmp_path: Path, *, exclude_processes: list[str] | None = None) -> AppConfig:
    data_dir = tmp_path / "data"
    return AppConfig(
        capture={"data_dir": data_dir, "staging_dir": data_dir / "staging"},
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
        embed={"text_model": "local-test"},
        qdrant={"enabled": False},
        tracking={"enabled": False},
        api={"require_api_key": True, "api_key": "secret", "bridge_token": "bridge"},
        privacy={"exclude_processes": exclude_processes or []},
        encryption={"enabled": False},
        security={"local_unlock_enabled": False},
    )


@pytest.mark.anyio
async def test_ingest_bridge_token_creates_capture(tmp_path: Path, async_client_factory) -> None:
    config = _build_config(tmp_path)
    app = create_app(config)
    async with async_client_factory(app) as client:
        metadata = {"app_name": "Chrome", "window_title": "Example", "monitor_id": "1"}
        files = {
            "metadata": (None, json.dumps(metadata), "application/json"),
            "image": ("shot.png", _sample_png_bytes(), "image/png"),
        }
        response = await client.post(
            "/api/events/ingest",
            files=files,
            headers={"X-Bridge-Token": "bridge"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"

        db = DatabaseManager(config.database)
        with db.session() as session:
            capture = session.get(CaptureRecord, payload["observation_id"])
        assert capture is not None
        assert capture.ocr_status == "pending"

        metadata["observation_id"] = payload["observation_id"]
        files = {
            "metadata": (None, json.dumps(metadata), "application/json"),
            "image": ("shot.png", _sample_png_bytes(), "image/png"),
        }
        repeat = await client.post(
            "/api/events/ingest", files=files, headers={"X-Bridge-Token": "bridge"}
        )
        assert repeat.status_code == 200
        assert repeat.json()["status"] == "exists"


@pytest.mark.anyio
async def test_ingest_respects_privacy_filter(tmp_path: Path, async_client_factory) -> None:
    config = _build_config(tmp_path, exclude_processes=["chrome"])
    app = create_app(config)
    metadata = {"app_name": "Chrome", "window_title": "Blocked", "monitor_id": "1"}
    files = {
        "metadata": (None, json.dumps(metadata), "application/json"),
        "image": ("shot.png", _sample_png_bytes(), "image/png"),
    }
    async with async_client_factory(app) as client:
        response = await client.post(
            "/api/events/ingest",
            files=files,
            headers={"X-Bridge-Token": "bridge"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "skipped"

    db = DatabaseManager(config.database)
    with db.session() as session:
        capture = session.get(CaptureRecord, payload["observation_id"])
    assert capture is not None
    assert capture.ocr_status == "skipped"


@pytest.mark.anyio
async def test_storage_endpoint_reports_usage(tmp_path: Path, async_client_factory) -> None:
    config = _build_config(tmp_path)
    media_dir = Path(config.capture.data_dir) / "media" / "roi"
    media_dir.mkdir(parents=True, exist_ok=True)
    sample_path = media_dir / "sample.bin"
    sample_path.write_bytes(b"12345")

    app = create_app(config)
    async with async_client_factory(app) as client:
        response = await client.get("/api/storage", headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["screenshot_ttl_days"] == config.retention.screenshot_ttl_days
    assert payload["media_usage_bytes"] >= sample_path.stat().st_size
