"""AES-GCM encryption helpers.

Important: For video segments / large media, avoid reading entire files into RAM.
This module uses streaming AES-GCM (Cipher + GCM) for large payloads.
Wire format:
  [12-byte nonce][ciphertext...][16-byte tag]
"""

from __future__ import annotations

import os
import sys
import ctypes
from ctypes import wintypes
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

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

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
        if provider == "windows-credential-manager":
            if sys.platform != "win32":  # pragma: no cover - platform guard
                raise RuntimeError("Windows Credential Manager is only available on Windows")
            key = _read_windows_credential(self._config.key_name)
            if key is None:
                key = os.urandom(32)
                _write_windows_credential(self._config.key_name, key)
            return _validate_key_length(key)
        if provider.startswith("file:"):
            path = Path(provider.split(":", 1)[1])
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                key = os.urandom(32)
                path.write_bytes(key)
            key = path.read_bytes()
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


def _read_windows_credential(target_name: str) -> bytes | None:  # pragma: no cover - Windows only
    class CREDENTIAL(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    cred_ptr = ctypes.POINTER(CREDENTIAL)()
    if not ctypes.windll.advapi32.CredReadW(target_name, 1, 0, ctypes.byref(cred_ptr)):
        return None
    try:
        size = cred_ptr.contents.CredentialBlobSize
        blob = ctypes.string_at(cred_ptr.contents.CredentialBlob, size)
        return bytes(blob)
    finally:
        ctypes.windll.advapi32.CredFree(cred_ptr)


def _write_windows_credential(target_name: str, key: bytes) -> None:  # pragma: no cover
    class CREDENTIAL(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    blob = (ctypes.c_byte * len(key)).from_buffer_copy(key)
    cred = CREDENTIAL()
    cred.Type = 1  # CRED_TYPE_GENERIC
    cred.TargetName = ctypes.c_wchar_p(target_name)
    cred.CredentialBlobSize = len(key)
    cred.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_byte))
    cred.Persist = 2  # CRED_PERSIST_LOCAL_MACHINE
    if not ctypes.windll.advapi32.CredWriteW(ctypes.byref(cred), 0):
        raise RuntimeError("Failed to write credential to Windows Credential Manager")
