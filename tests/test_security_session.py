from __future__ import annotations

import pytest

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig


@pytest.mark.anyio
async def test_unlock_required_for_protected_endpoints(
    tmp_path, monkeypatch, async_client_factory
) -> None:
    monkeypatch.setenv("AUTOCAPTURE_TEST_MODE", "0")
    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        qdrant={"enabled": False},
        embed={"text_model": "local-test"},
        tracking={"enabled": False},
        security={"local_unlock_enabled": True, "provider": "test"},
    )
    app = create_app(config)

    async with async_client_factory(app) as client:
        response = await client.post("/api/retrieve", json={"query": "hello", "k": 1})
        assert response.status_code == 401

        unlock = await client.post("/api/unlock")
        assert unlock.status_code == 200
        token = unlock.json().get("token")
        assert token

        response = await client.post(
            "/api/retrieve",
            json={"query": "hello", "k": 1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
