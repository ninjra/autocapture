from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.plugins import PluginManager
from autocapture.plugins.errors import PluginResolutionError
from autocapture.plugins.settings import update_plugin_settings

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "plugins"


def _make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
    )


def _copy_fixture(name: str, dest_root: Path) -> Path:
    src = FIXTURES / name
    dest = dest_root / name
    shutil.copytree(src, dest)
    return dest


def test_directory_discovery_does_not_import_disabled(tmp_path: Path, monkeypatch) -> None:
    plugin_root = tmp_path / "plugins"
    plugin_root.mkdir()
    _copy_fixture("poison", plugin_root)
    sentinel = tmp_path / "sentinel.txt"
    monkeypatch.setenv("POISON_SENTINEL", str(sentinel))
    config = _make_config(tmp_path)
    manager = PluginManager(
        config,
        settings_path=tmp_path / "settings.json",
        plugin_dir=plugin_root,
    )
    assert not sentinel.exists()
    manager.enable_plugin("test.poison", accept_hashes=True)
    _ = manager.resolve_extension("embedder.text", "poison-embedder", use_cache=False)
    assert sentinel.exists()


def test_lock_mismatch_blocks_plugin(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugins"
    plugin_root.mkdir()
    plugin_dir = _copy_fixture("sample", plugin_root)
    config = _make_config(tmp_path)
    manager = PluginManager(
        config,
        settings_path=tmp_path / "settings.json",
        plugin_dir=plugin_root,
    )
    status = manager.enable_plugin("test.sample", accept_hashes=True)
    assert status.enabled
    module_path = plugin_dir / "plugin_module.py"
    module_path.write_text(module_path.read_text(encoding="utf-8") + "\n# changed\n")
    manager.refresh()
    status = next(entry for entry in manager.catalog() if entry.plugin.plugin_id == "test.sample")
    assert status.blocked
    assert status.reason == "lock_mismatch"
    with pytest.raises(PluginResolutionError):
        manager.resolve_record("embedder.text", "test-embedder")


def test_extension_conflict_requires_override(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugins"
    plugin_root.mkdir()
    plugin_a = plugin_root / "conflict_a"
    plugin_b = plugin_root / "conflict_b"
    plugin_a.mkdir()
    plugin_b.mkdir()
    manifest = """
schema_version: 1
plugin_id: {plugin_id}
name: Conflict Plugin
version: 0.1.0
enabled_by_default: false
extensions:
  - kind: embedder.text
    id: conflict-embedder
    name: Conflict Embedder
    factory:
      type: python
      entrypoint: tests.fixtures.plugins.sample.plugin_module:create_embedder
"""
    (plugin_a / "plugin.yaml").write_text(manifest.format(plugin_id="test.conflict.a"))
    (plugin_b / "plugin.yaml").write_text(manifest.format(plugin_id="test.conflict.b"))
    config = _make_config(tmp_path)
    settings_path = tmp_path / "settings.json"
    manager = PluginManager(config, settings_path=settings_path, plugin_dir=plugin_root)
    manager.enable_plugin("test.conflict.a", accept_hashes=True)
    manager.enable_plugin("test.conflict.b", accept_hashes=True)
    with pytest.raises(PluginResolutionError):
        manager.resolve_record("embedder.text", "conflict-embedder")

    def _apply(settings):
        settings.extension_overrides["embedder.text:conflict-embedder"] = "test.conflict.a"
        return settings

    update_plugin_settings(settings_path, _apply)
    manager.refresh()
    record = manager.resolve_record("embedder.text", "conflict-embedder")
    assert record.plugin_id == "test.conflict.a"
