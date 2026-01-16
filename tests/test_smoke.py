import importlib.util

from autocapture.config import AppConfig, CaptureConfig, DatabaseConfig, WorkerConfig
from autocapture.smoke import run_smoke


def test_smoke_runs(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    original = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name in {"fastembed", "sentence_transformers"}:
            return None
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'smoke.db'}"),
        capture=CaptureConfig(staging_dir=tmp_path / "staging", data_dir=tmp_path),
        worker=WorkerConfig(data_dir=tmp_path),
    )
    assert run_smoke(config) == 0
