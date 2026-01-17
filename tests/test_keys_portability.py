from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import pytest

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.security.portable_keys import (
    _read_dpapi_file,
    _write_dpapi_file,
    export_keys,
    import_keys,
)


def _write_key(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_export_import_roundtrip(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    secrets_dir = data_dir / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)

    pseudonym_key = b"p" * 32
    token_key = b"t" * 32
    sqlcipher_key = b"s" * 32
    tracking_key = b"h" * 32
    media_key = b"m" * 32

    _write_key(secrets_dir / "pseudonym.key", pseudonym_key)
    _write_key(secrets_dir / "token_vault.key", token_key)
    _write_key(secrets_dir / "sqlcipher.key", sqlcipher_key)
    _write_key(secrets_dir / "host_events.key", tracking_key)
    _write_key(secrets_dir / "media.key", media_key)

    config = AppConfig(
        capture={"data_dir": data_dir, "staging_dir": data_dir / "staging"},
        database=DatabaseConfig(
            url="sqlite:///:memory:",
            encryption_enabled=True,
            encryption_provider="file",
            encryption_key_path=Path("secrets/sqlcipher.key"),
        ),
        tracking={
            "enabled": True,
            "db_path": Path("host_events.sqlite"),
            "encryption_enabled": True,
            "encryption_key_provider": "file",
            "encryption_key_path": Path("secrets/host_events.key"),
        },
        encryption={
            "enabled": True,
            "key_provider": f"file:{(secrets_dir / 'media.key').resolve().as_posix()}",
        },
        privacy={"token_vault_enabled": True},
    )

    bundle_path = tmp_path / "keys.json"
    export_keys(config, bundle_path, "correct-horse-battery-staple")

    for path in [
        secrets_dir / "pseudonym.key",
        secrets_dir / "token_vault.key",
        secrets_dir / "sqlcipher.key",
        secrets_dir / "host_events.key",
        secrets_dir / "media.key",
    ]:
        path.unlink()

    import_keys(config, bundle_path, "correct-horse-battery-staple")

    use_dpapi = os.name == "nt"
    assert _read_dpapi_file(secrets_dir / "pseudonym.key", use_dpapi=use_dpapi) == pseudonym_key
    assert _read_dpapi_file(secrets_dir / "token_vault.key", use_dpapi=use_dpapi) == token_key
    assert (secrets_dir / "sqlcipher.key").read_bytes() == sqlcipher_key
    assert (secrets_dir / "host_events.key").read_bytes() == tracking_key
    assert (secrets_dir / "media.key").read_bytes() == media_key


def test_import_rejects_wrong_password(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    secrets_dir = data_dir / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    _write_key(secrets_dir / "pseudonym.key", b"p" * 32)

    config = AppConfig(
        capture={"data_dir": data_dir, "staging_dir": data_dir / "staging"},
    )

    bundle_path = tmp_path / "keys.json"
    export_keys(config, bundle_path, "correct-horse-battery-staple")

    with pytest.raises(RuntimeError, match="Invalid password"):
        import_keys(config, bundle_path, "wrong-password")


def test_dpapi_helpers_use_win32crypt(monkeypatch, tmp_path: Path) -> None:
    stub = types.SimpleNamespace(
        CryptProtectData=lambda data, *_args: b"protected:" + data,
        CryptUnprotectData=lambda data, *_args: (None, data.replace(b"protected:", b"", 1)),
    )
    monkeypatch.setitem(sys.modules, "win32crypt", stub)
    path = tmp_path / "dpapi.key"
    _write_dpapi_file(path, b"secret", use_dpapi=True)
    assert path.read_bytes() == b"protected:secret"
    assert _read_dpapi_file(path, use_dpapi=True) == b"secret"
