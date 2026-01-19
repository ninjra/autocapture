from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.plugins import PluginManager

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "plugins"


@pytest.mark.anyio
async def test_plugins_catalog_does_not_import_disabled(
    tmp_path: Path, async_client_factory, monkeypatch
) -> None:
    plugin_root = tmp_path / "plugins"
    plugin_root.mkdir()
    shutil.copytree(FIXTURES / "poison", plugin_root / "poison")
    sentinel = tmp_path / "sentinel.txt"
    monkeypatch.setenv("POISON_SENTINEL", str(sentinel))

    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
    )
    plugins = PluginManager(
        config,
        settings_path=tmp_path / "settings.json",
        plugin_dir=plugin_root,
    )
    app = create_app(config, plugin_manager=plugins)
    async with async_client_factory(app) as client:
        response = await client.get("/api/plugins/catalog")
        assert response.status_code == 200
        response = await client.get("/api/plugins/extensions", params={"kind": "embedder.text"})
        assert response.status_code == 200
    assert not sentinel.exists()
