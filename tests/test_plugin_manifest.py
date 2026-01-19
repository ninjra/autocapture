from __future__ import annotations

import pytest

from autocapture.plugins.errors import PluginManifestError
from autocapture.plugins.manifest import parse_manifest


def test_manifest_parse_valid() -> None:
    payload = {
        "schema_version": 1,
        "plugin_id": "test.sample",
        "name": "Sample",
        "version": "0.1.0",
        "extensions": [
            {
                "kind": "embedder.text",
                "id": "sample-embedder",
                "name": "Sample Embedder",
                "factory": {
                    "type": "python",
                    "entrypoint": "tests.fixtures.plugins.sample.plugin_module:create_embedder",
                },
            }
        ],
    }
    manifest = parse_manifest(payload)
    assert manifest.plugin_id == "test.sample"
    assert manifest.extensions[0].kind == "embedder.text"


def test_manifest_rejects_unknown_schema() -> None:
    with pytest.raises(PluginManifestError):
        parse_manifest(
            {"schema_version": 2, "plugin_id": "test.bad", "name": "bad", "version": "0"}
        )


def test_manifest_rejects_duplicate_extensions() -> None:
    payload = {
        "schema_version": 1,
        "plugin_id": "test.dupe",
        "name": "Duplicate",
        "version": "0.1.0",
        "extensions": [
            {
                "kind": "embedder.text",
                "id": "dupe",
                "name": "Dup1",
                "factory": {
                    "type": "python",
                    "entrypoint": "tests.fixtures.plugins.sample.plugin_module:create_embedder",
                },
            },
            {
                "kind": "embedder.text",
                "id": "dupe",
                "name": "Dup2",
                "factory": {
                    "type": "python",
                    "entrypoint": "tests.fixtures.plugins.sample.plugin_module:create_embedder",
                },
            },
        ],
    }
    with pytest.raises(PluginManifestError):
        parse_manifest(payload)


def test_manifest_rejects_invalid_ids() -> None:
    payload = {
        "schema_version": 1,
        "plugin_id": "INVALID",
        "name": "bad",
        "version": "0.1.0",
        "extensions": [],
    }
    with pytest.raises(PluginManifestError):
        parse_manifest(payload)
