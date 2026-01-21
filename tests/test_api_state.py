from __future__ import annotations

import pytest

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig


@pytest.mark.anyio
async def test_api_state_snapshot(tmp_path, async_client_factory) -> None:
    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
    )
    app = create_app(config)
    async with async_client_factory(app) as client:
        response = await client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == 1
    assert "health" in data
    assert "storage" in data
