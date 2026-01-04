from __future__ import annotations

import os
from pathlib import Path

from autocapture.config import EncryptionConfig
from autocapture.encryption import EncryptionManager


def test_file_key_provider_roundtrip(tmp_path: Path) -> None:
    key_path = tmp_path / "key.bin"
    key_path.write_bytes(os.urandom(32))
    config = EncryptionConfig(enabled=True, key_provider=f"file:{key_path}")
    manager = EncryptionManager(config)

    source = tmp_path / "source.txt"
    source.write_text("secret", encoding="utf-8")
    encrypted = tmp_path / "encrypted.bin"
    decrypted = tmp_path / "decrypted.txt"

    manager.encrypt_file(source, encrypted)
    manager.decrypt_file(encrypted, decrypted)
    assert decrypted.read_text(encoding="utf-8") == "secret"


def test_env_key_provider_roundtrip(tmp_path: Path, monkeypatch) -> None:
    key = os.urandom(32)
    monkeypatch.setenv("AUTOCAPTURE_TEST_KEY", key.hex())
    config = EncryptionConfig(enabled=True, key_provider="env:AUTOCAPTURE_TEST_KEY")
    manager = EncryptionManager(config)

    source = tmp_path / "source.txt"
    source.write_text("secret", encoding="utf-8")
    encrypted = tmp_path / "encrypted.bin"
    decrypted = tmp_path / "decrypted.txt"

    manager.encrypt_file(source, encrypted)
    manager.decrypt_file(encrypted, decrypted)
    assert decrypted.read_text(encoding="utf-8") == "secret"
