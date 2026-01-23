from __future__ import annotations

from pathlib import Path

import pytest

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig


def _make_app(tmp_path: Path):
    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
    )
    app = create_app(config)
    return app


@pytest.mark.anyio
async def test_api_storage_stats(tmp_path: Path, async_client_factory) -> None:
    app = _make_app(tmp_path)
    async with async_client_factory(app) as client:
        response = await client.get("/api/storage/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["data_dir"]
    assert "media_bytes" in payload
    assert "total_bytes" in payload
