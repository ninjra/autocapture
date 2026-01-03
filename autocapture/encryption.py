"""AES-GCM encryption helpers.

Important: For video segments / large media, avoid reading entire files into RAM.
This module uses streaming AES-GCM (Cipher + GCM) for large payloads.
Wire format:
  [12-byte nonce][ciphertext...][16-byte tag]
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .config import EncryptionConfig
from .logging_utils import get_logger


class EncryptionManager:
    """Encrypt/decrypt files before transferring to NAS."""

    def __init__(self, config: EncryptionConfig) -> None:
        self._config = config
        self._log = get_logger("encryption")
        self._key: bytes | None = None
        if self._config.enabled:
            self._key = self._load_key()

    def encrypt_file(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not self._config.enabled:
            with source.open("rb") as src, destination.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            return

        nonce = os.urandom(12)
        encryptor = Cipher(algorithms.AES(self._key), modes.GCM(nonce)).encryptor()
        with source.open("rb") as src, destination.open("wb") as dst:
            dst.write(nonce)
            while True:
                chunk = src.read(1024 * 1024)  # 1 MiB
                if not chunk:
                    break
                dst.write(encryptor.update(chunk))
            dst.write(encryptor.finalize())
            dst.write(encryptor.tag)
        self._log.debug("Encrypted %s -> %s", source, destination)

    def decrypt_file(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not self._config.enabled:
            with source.open("rb") as src, destination.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            return

        # AES-GCM needs the final tag up front to initialize the decryptor.
        # We do a cheap seek to read the last 16 bytes (tag), then stream-decrypt.
        tag_len = 16
        with source.open("rb") as src, destination.open("wb") as dst:
            nonce = src.read(12)
            if len(nonce) != 12:
                raise RuntimeError("Encrypted file missing nonce header")
            src.seek(0, 2)
            size = src.tell()
            if size < 12 + tag_len:
                raise RuntimeError("Encrypted file too small to contain tag")
            src.seek(size - tag_len)
            tag = src.read(tag_len)
            src.seek(12)

            decryptor = Cipher(algorithms.AES(self._key), modes.GCM(nonce, tag)).decryptor()
            remaining = (size - tag_len) - 12
            while remaining > 0:
                chunk = src.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                dst.write(decryptor.update(chunk))
            dst.write(decryptor.finalize())

    def _load_key(self) -> bytes:
        provider = self._config.key_provider
        if (
            provider == "windows-credential-manager"
        ):  # pragma: no cover - Windows specific
            import win32cred

            cred = win32cred.CredRead(
                self._config.key_name, win32cred.CRED_TYPE_GENERIC
            )
            key = cred["CredentialBlob"]
            return _validate_key_length(key)
        if provider.startswith("file:"):
            key = Path(provider.split(":", 1)[1]).read_bytes()
            return _validate_key_length(key)
        if provider.startswith("env:"):
            key = os.getenv(provider.split(":", 1)[1])
            if not key:
                raise RuntimeError("Encryption key environment variable is missing")
            return _validate_key_length(bytes.fromhex(key))
        raise ValueError(f"Unsupported key provider: {provider}")


def _validate_key_length(key: bytes) -> bytes:
    if len(key) not in (16, 24, 32):
        raise ValueError("AES key must be 16, 24, or 32 bytes long")
    return key
