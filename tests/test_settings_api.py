from __future__ import annotations

from pathlib import Path

import pytest

from autocapture.api.server import create_app
import datetime as dt

from autocapture.config import AppConfig, DatabaseConfig, apply_settings_overrides


@pytest.mark.anyio
async def test_settings_persist_and_apply(tmp_path: Path, async_client_factory) -> None:
    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
    )
    app = create_app(config)
    snooze_until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    payload = {
        "settings": {
            "routing": {
                "ocr": "local",
                "embedding": "local",
                "retrieval": "local",
                "compressor": "extractive",
                "verifier": "rules",
                "llm": "openai",
            },
            "privacy": {
                "paused": True,
                "snooze_until_utc": snooze_until.isoformat(),
            },
        }
    }
    async with async_client_factory(app) as client:
        response = await client.post("/api/settings", json=payload)
    assert response.status_code == 200
    assert config.routing.llm == "openai"
    assert config.privacy.paused is True

    settings_path = tmp_path / "settings.json"
    assert settings_path.exists()

    fresh_config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
    )
    updated = apply_settings_overrides(fresh_config)
    assert updated.routing.llm == "openai"
    assert updated.privacy.paused is True
    assert updated.privacy.snooze_until_utc == snooze_until
