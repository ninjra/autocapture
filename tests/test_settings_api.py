from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig, apply_settings_overrides


def test_settings_persist_and_apply(tmp_path: Path) -> None:
    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
    )
    app = create_app(config)
    client = TestClient(app)
    payload = {
        "settings": {
            "routing": {
                "ocr": "local",
                "embedding": "local",
                "retrieval": "local",
                "compressor": "extractive",
                "verifier": "rules",
                "llm": "openai",
            }
        }
    }
    response = client.post("/api/settings", json=payload)
    assert response.status_code == 200
    assert config.routing.llm == "openai"

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
