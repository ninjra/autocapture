from __future__ import annotations

import pytest

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig


@pytest.mark.anyio
async def test_healthz_deep_ok_when_dependencies_skipped(tmp_path, async_client_factory) -> None:
    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        qdrant={"enabled": False},
        embed={"text_model": "local-test"},
        tracking={"enabled": False},
    )
    app = create_app(config)
    async with async_client_factory(app) as client:
        response = await client.get("/healthz/deep")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["checks"]["db"]["ok"] is True
    assert data["checks"]["qdrant"]["skipped"] is True


@pytest.mark.anyio
async def test_healthz_deep_reports_db_failure(tmp_path, monkeypatch, async_client_factory) -> None:
    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        qdrant={"enabled": False},
        embed={"text_model": "local-test"},
        tracking={"enabled": False},
    )
    app = create_app(config)

    def fail_session():
        raise RuntimeError("db down")

    monkeypatch.setattr(app.state.db, "session", fail_session)
    async with async_client_factory(app) as client:
        response = await client.get("/healthz/deep")
    assert response.status_code == 503
    data = response.json()
    assert data["ok"] is False
    assert data["checks"]["db"]["ok"] is False
