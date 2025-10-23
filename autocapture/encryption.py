"""AES-GCM streaming encryption helpers."""

from __future__ import annotations

import os
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import EncryptionConfig
from .logging_utils import get_logger


class EncryptionManager:
    """Encrypt/decrypt files before transferring to NAS."""

    def __init__(self, config: EncryptionConfig) -> None:
        self._config = config
        self._log = get_logger("encryption")
        self._key = self._load_key()

    def encrypt_file(self, source: Path, destination: Path) -> None:
        if not self._config.enabled:
            destination.write_bytes(source.read_bytes())
            return

        nonce = os.urandom(12)
        cipher = AESGCM(self._key)
        data = source.read_bytes()
        encrypted = cipher.encrypt(nonce, data, None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(nonce + encrypted)
        self._log.debug("Encrypted %s -> %s", source, destination)

    def decrypt_file(self, source: Path, destination: Path) -> None:
        if not self._config.enabled:
            destination.write_bytes(source.read_bytes())
            return

        raw = source.read_bytes()
        nonce, ciphertext = raw[:12], raw[12:]
        cipher = AESGCM(self._key)
        decrypted = cipher.decrypt(nonce, ciphertext, None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(decrypted)

    def _load_key(self) -> bytes:
        provider = self._config.key_provider
        if (
            provider == "windows-credential-manager"
        ):  # pragma: no cover - Windows specific
            import win32cred

            cred = win32cred.CredRead(
                self._config.key_name, win32cred.CRED_TYPE_GENERIC
            )
            return cred["CredentialBlob"]
        if provider.startswith("file:"):
            return Path(provider.split(":", 1)[1]).read_bytes()
        if provider.startswith("env:"):
            key = os.getenv(provider.split(":", 1)[1])
            if not key:
                raise RuntimeError("Encryption key environment variable is missing")
            return bytes.fromhex(key)
        raise ValueError(f"Unsupported key provider: {provider}")
