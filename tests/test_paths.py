from __future__ import annotations

from pathlib import Path

from autocapture import paths


def test_windows_local_appdata_falls_back_to_userprofile(monkeypatch) -> None:
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    monkeypatch.setenv("USERPROFILE", r"C:\Users\Casey")

    resolved = paths.app_local_data_dir()
    normalized = str(resolved).replace("\\", "/")

    assert normalized.endswith("/AppData/Local/Autocapture")
    assert "C:" in normalized
    assert "/home/" not in normalized


def test_doctor_path_failure_includes_env_hint(monkeypatch, tmp_path: Path) -> None:
    from autocapture import doctor
    from autocapture.config import AppConfig

    config = AppConfig()
    config.capture.data_dir = tmp_path / "blocked-data"
    config.capture.staging_dir = tmp_path / "blocked-staging"

    monkeypatch.setattr(doctor.sys, "platform", "win32")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("USERPROFILE", r"C:\Users\Test")
    monkeypatch.setenv("APPDATA", r"C:\Users\Test\AppData\Roaming")
    monkeypatch.setenv("HOMEDRIVE", "C:")
    monkeypatch.setenv("HOMEPATH", r"\Users\Test")

    original_mkdir = Path.mkdir
    original_write_text = Path.write_text

    def _mkdir(self: Path, *args, **kwargs):
        if "blocked" in str(self):
            raise PermissionError("blocked")
        return original_mkdir(self, *args, **kwargs)

    def _write_text(self: Path, *args, **kwargs):
        if "blocked" in str(self):
            raise PermissionError("blocked")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _mkdir)
    monkeypatch.setattr(Path, "write_text", _write_text)

    result = doctor._check_paths(config)

    assert not result.ok
    assert "path=" in result.detail
    assert "missing_env=LOCALAPPDATA" in result.detail
    assert "set LOCALAPPDATA" in result.detail
