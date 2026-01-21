from __future__ import annotations

from pathlib import Path

from autocapture.config import AppConfig
from autocapture.qdrant.sidecar import QdrantSidecar, should_manage_sidecar


def _base_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        database={"url": "sqlite:///:memory:"},
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
        capture={
            "data_dir": tmp_path,
            "staging_dir": tmp_path / "staging",
        },
        worker={"data_dir": tmp_path},
    )


def test_should_manage_sidecar_localhost(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.routing.vector_backend = "qdrant"
    config.qdrant.url = "http://127.0.0.1:6333"
    assert should_manage_sidecar(config) is True


def test_should_manage_sidecar_remote(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.routing.vector_backend = "qdrant"
    config.qdrant.url = "http://qdrant.example.com:6333"
    assert should_manage_sidecar(config) is False


def test_sidecar_missing_binary_no_crash(tmp_path: Path, monkeypatch) -> None:
    config = _base_config(tmp_path)
    config.routing.vector_backend = "qdrant"
    config.qdrant.url = "http://127.0.0.1:6333"
    monkeypatch.setattr(
        "autocapture.qdrant.sidecar.resolve_qdrant_path",
        lambda _config: None,
    )
    sidecar = QdrantSidecar(config, tmp_path, tmp_path / "logs")
    sidecar.start()
    sidecar.stop()
