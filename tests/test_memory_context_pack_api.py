from __future__ import annotations

from pathlib import Path

import pytest

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.memory.models import ArtifactMeta
from autocapture.memory.store import MemoryStore


@pytest.mark.anyio
async def test_context_pack_includes_memory_snapshot(
    tmp_path: Path, async_client_factory
) -> None:
    memory_root = tmp_path / "memory"
    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
        memory={
            "enabled": True,
            "api_context_pack_enabled": True,
            "storage": {"root_dir": memory_root},
        },
    )

    store = MemoryStore(config.memory)
    store.ingest_text(
        "Alpha\n\nBeta Gamma",
        ArtifactMeta(source_uri="stdin", title="Memory Doc"),
    )

    app = create_app(config)
    async with async_client_factory(app) as client:
        response = await client.post(
            "/api/context-pack",
            json={"query": "Beta", "pack_format": "json", "include_memory_snapshot": True},
        )
    assert response.status_code == 200
    payload = response.json()
    snapshot = payload.get("memory_snapshot")
    assert snapshot is not None
    assert "result" in snapshot
    assert "context_md" in snapshot
    assert "citations" in snapshot
    assert snapshot["result"].get("snapshot_id")
