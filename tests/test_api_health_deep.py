from __future__ import annotations

from fastapi.testclient import TestClient

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig


def test_healthz_deep_ok_when_dependencies_skipped(tmp_path) -> None:
    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        qdrant={"enabled": False},
        embed={"text_model": "local-test"},
        tracking={"enabled": False},
    )
    app = create_app(config)
    client = TestClient(app)
    response = client.get("/healthz/deep")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["checks"]["db"]["ok"] is True
    assert data["checks"]["qdrant"]["skipped"] is True


def test_healthz_deep_reports_db_failure(tmp_path, monkeypatch) -> None:
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
    client = TestClient(app)
    response = client.get("/healthz/deep")
    assert response.status_code == 503
    data = response.json()
    assert data["ok"] is False
    assert data["checks"]["db"]["ok"] is False
