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
async def test_api_plugins_schema_detail_preview_apply(tmp_path: Path, async_client_factory) -> None:
    app = _make_app(tmp_path)
    async with async_client_factory(app) as client:
        schema_resp = await client.get("/api/plugins/schema")
        assert schema_resp.status_code == 200
        schema_payload = schema_resp.json()
        assert schema_payload["plugins"]
        plugin_id = schema_payload["plugins"][0]["plugin_id"]

        detail_resp = await client.get(f"/api/plugins/{plugin_id}")
        assert detail_resp.status_code == 200
        detail_payload = detail_resp.json()
        assert detail_payload["plugin_id"] == plugin_id

        preview_resp = await client.post(
            f"/api/plugins/{plugin_id}/preview",
            json={"candidate": {"sample": "value"}},
        )
        assert preview_resp.status_code == 200
        preview_payload = preview_resp.json()
        assert preview_payload["preview_id"]

        apply_resp = await client.post(
            f"/api/plugins/{plugin_id}/apply",
            json={"candidate": {"sample": "value"}, "preview_id": preview_payload["preview_id"]},
        )
        assert apply_resp.status_code == 200
        apply_payload = apply_resp.json()
        assert apply_payload["status"] == "ok"

        detail_resp = await client.get(f"/api/plugins/{plugin_id}")
        assert detail_resp.status_code == 200
        detail_payload = detail_resp.json()
        assert detail_payload["config"].get("sample") == "value"
