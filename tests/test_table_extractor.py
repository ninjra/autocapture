from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sqlalchemy import text

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.enrichment.table_extractor import TableExtractionService
from autocapture.indexing.sqlite_utils import vector_to_blob
from autocapture.plugins import PluginManager
from autocapture.plugins.settings import update_plugin_settings
from autocapture.storage.database import DatabaseManager

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "plugins"


class _StubEmbedder:
    def __init__(self) -> None:
        self.model_name = "stub"
        self.dim = 3

    def embed_texts(self, texts):
        return [[float(len(text)), 0.0, 1.0] for text in texts]


def _make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'table.db'}", sqlite_wal=False),
        tracking={"enabled": False},
    )


def _copy_fixture(name: str, dest_root: Path) -> Path:
    src = FIXTURES / name
    dest = dest_root / name
    shutil.copytree(src, dest)
    return dest


def _setup_manager(tmp_path: Path, config: AppConfig) -> tuple[PluginManager, Path]:
    plugin_root = tmp_path / "plugins"
    plugin_root.mkdir()
    _copy_fixture("table_extractor", plugin_root)
    settings_path = tmp_path / "settings.json"
    manager = PluginManager(config, settings_path=settings_path, plugin_dir=plugin_root)
    manager.enable_plugin("test.table_extractor", accept_hashes=True)
    return manager, settings_path


def _write_cloud_plugin(plugin_root: Path) -> None:
    plugin_dir = plugin_root / "cloud_table"
    plugin_dir.mkdir()
    manifest = """
schema_version: 1
plugin_id: test.table_extractor_cloud
name: Test Table Extractor Cloud
version: 0.1.0
enabled_by_default: false
extensions:
  - kind: table.extractor
    id: cloud-table
    name: Cloud Table Extractor
    aliases: []
    pillars:
      data_handling:
        cloud: required
        cloud_images: none
        supports_local: false
    factory:
      type: python
      entrypoint: tests.fixtures.plugins.table_extractor.plugin_module:create_extractor
""".lstrip()
    (plugin_dir / "plugin.yaml").write_text(manifest, encoding="utf-8")


def test_table_extractor_inserts_and_auto_embeds(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.table_extractor.enabled = True
    config.routing.table_extractor = "test-table"
    manager, settings_path = _setup_manager(tmp_path, config)

    def _apply(settings):
        settings.configs["test.table_extractor"] = {
            "rows": [{"id": "row-1", "note": "hello"}],
        }
        return settings

    update_plugin_settings(settings_path, _apply)
    manager.refresh()
    db = DatabaseManager(config.database)
    service = TableExtractionService(
        config,
        db,
        embedder=_StubEmbedder(),
        plugin_manager=manager,
    )
    outcome = service.extract_and_store()
    assert outcome.status == "ok"
    assert outcome.inserted == 1
    with db.engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, note, note_embedding FROM extracted_table")
        ).fetchone()
    assert row is not None
    blob = row[2].tobytes() if isinstance(row[2], memoryview) else row[2]
    assert blob == vector_to_blob([5.0, 0.0, 1.0])


def test_table_extractor_policy_gate_blocks(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.table_extractor.enabled = True
    config.table_extractor.allow_cloud = False
    config.routing.table_extractor = "cloud-table"
    plugin_root = tmp_path / "plugins"
    plugin_root.mkdir()
    _write_cloud_plugin(plugin_root)
    settings_path = tmp_path / "settings.json"
    manager = PluginManager(config, settings_path=settings_path, plugin_dir=plugin_root)
    manager.enable_plugin("test.table_extractor_cloud", accept_hashes=True)
    manager.refresh()
    db = DatabaseManager(config.database)
    service = TableExtractionService(
        config,
        db,
        embedder=_StubEmbedder(),
        plugin_manager=manager,
    )
    outcome = service.extract_and_store()
    assert outcome.status == "policy_blocked"
    with db.engine.begin() as conn:
        exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='extracted_table'")
        ).fetchone()
    assert exists is None


def test_table_extractor_rolls_back_on_failure(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.table_extractor.enabled = True
    config.routing.table_extractor = "test-table"
    manager, settings_path = _setup_manager(tmp_path, config)

    def _apply(settings):
        settings.configs["test.table_extractor"] = {
            "table": {
                "name": "extracted_json",
                "columns": [
                    {"name": "id", "dtype": "text", "nullable": False},
                    {"name": "payload", "dtype": "json", "nullable": False},
                ],
                "primary_key": ["id"],
            },
            "rows": [
                {"id": "row-1", "payload": {"ok": True}},
                {"id": "row-2", "payload": {"__bad__": True}},
            ],
        }
        return settings

    update_plugin_settings(settings_path, _apply)
    manager.refresh()
    db = DatabaseManager(config.database)
    service = TableExtractionService(
        config,
        db,
        embedder=_StubEmbedder(),
        plugin_manager=manager,
    )
    with pytest.raises(Exception):
        service.extract_and_store()
    with db.engine.begin() as conn:
        exists = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='extracted_json'")
        ).fetchone()
        if exists:
            count = conn.execute(text("SELECT COUNT(*) FROM extracted_json")).scalar()
            assert count == 0
