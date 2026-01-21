from __future__ import annotations

import json
from pathlib import Path

import pytest

from autocapture.config import AppConfig
from autocapture.security.extensions import guarded_load_extension, sha256_file


class _DummyConn:
    def __init__(self) -> None:
        self.enabled_flags: list[bool] = []
        self.loaded: list[str] = []

    def enable_load_extension(self, flag: bool) -> None:
        self.enabled_flags.append(bool(flag))

    def load_extension(self, path: str) -> None:
        self.loaded.append(path)


def _write_checksums(path: Path, entries: list[dict]) -> None:
    payload = {"version": 1, "files": entries}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_guarded_load_extension_fails_closed_secure_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ext_path = tmp_path / "ext.bin"
    ext_path.write_bytes(b"example")
    _write_checksums(tmp_path / "CHECKSUMS.json", [])
    config = AppConfig()
    config.security.secure_mode = True
    conn = _DummyConn()
    with pytest.raises(RuntimeError):
        guarded_load_extension(conn, ext_path, config=config)
    assert conn.loaded == []


def test_guarded_load_extension_allows_verified(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ext_path = tmp_path / "ext.bin"
    ext_path.write_bytes(b"example")
    digest = sha256_file(ext_path)
    _write_checksums(
        tmp_path / "CHECKSUMS.json", [{"path": ext_path.name, "sha256": digest}]
    )
    config = AppConfig()
    config.security.secure_mode = True
    conn = _DummyConn()
    assert guarded_load_extension(conn, ext_path, config=config) is True
    assert conn.loaded == [str(ext_path)]


def test_guarded_load_extension_warns_in_insecure_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ext_path = tmp_path / "ext.bin"
    ext_path.write_bytes(b"example")
    _write_checksums(tmp_path / "CHECKSUMS.json", [])
    config = AppConfig()
    config.security.secure_mode = False
    conn = _DummyConn()
    assert guarded_load_extension(conn, ext_path, config=config) is False
    assert conn.loaded == []
