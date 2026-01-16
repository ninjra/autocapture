"""Portable encrypted key export/import helpers."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..config import AppConfig, DatabaseConfig, EncryptionConfig, TrackingConfig
from ..encryption import _read_windows_credential, _write_windows_credential
from ..logging_utils import get_logger
from ..memory.entities import SecretStore
from .sqlcipher import load_sqlcipher_key
from .token_vault import TokenVaultKeyStore

_VERSION = 1
_KDF_N = 2**15
_KDF_R = 8
_KDF_P = 1
_KDF_LEN = 32
_KDF_MAXMEM = 64 * 1024 * 1024

_KEY_PSEUDONYM = "pseudonym_key"
_KEY_TOKEN_VAULT = "token_vault_key"
_KEY_SQLCIPHER = "sqlcipher_key"
_KEY_TRACKING = "tracking_sqlcipher_key"
_KEY_MEDIA = "media_encryption_key"


def export_keys(config: AppConfig, out_path: Path, password: str) -> None:
    _require_password(password)
    keys = _collect_keys(config)
    payload = {"keys": {name: _b64encode(value) for name, value in keys.items()}}
    plaintext = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

    salt = os.urandom(16)
    kek = _derive_key(password, salt)
    nonce = os.urandom(12)
    ciphertext = AESGCM(kek).encrypt(nonce, plaintext, None)

    envelope = {
        "version": _VERSION,
        "kdf": {
            "name": "scrypt",
            "n": _KDF_N,
            "r": _KDF_R,
            "p": _KDF_P,
            "salt": _b64encode(salt),
            "maxmem": _KDF_MAXMEM,
        },
        "aead": {
            "name": "aes-256-gcm",
            "nonce": _b64encode(nonce),
            "ciphertext": _b64encode(ciphertext),
        },
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")


def import_keys(config: AppConfig, in_path: Path, password: str) -> None:
    _require_password(password)
    envelope = json.loads(in_path.read_text(encoding="utf-8"))
    if envelope.get("version") != _VERSION:
        raise RuntimeError("Unsupported key bundle version")
    kdf = envelope.get("kdf", {})
    if kdf.get("name") != "scrypt":
        raise RuntimeError("Unsupported key derivation function")
    aead = envelope.get("aead", {})
    try:
        salt = _b64decode(kdf["salt"])
        nonce = _b64decode(aead["nonce"])
        ciphertext = _b64decode(aead["ciphertext"])
    except Exception as exc:
        raise RuntimeError("Invalid key bundle encoding") from exc

    kek = _derive_key(password, salt)
    try:
        plaintext = AESGCM(kek).decrypt(nonce, ciphertext, None)
    except Exception as exc:
        raise RuntimeError("Invalid password or corrupted key bundle") from exc

    payload = json.loads(plaintext.decode("utf-8"))
    keys = payload.get("keys", {})
    if not isinstance(keys, dict):
        raise RuntimeError("Invalid key bundle payload")

    _store_keys(config, {name: _b64decode(value) for name, value in keys.items()})


def _collect_keys(config: AppConfig) -> dict[str, bytes]:
    data_dir = Path(config.capture.data_dir)
    keys: dict[str, bytes] = {}

    secret_store = SecretStore(data_dir)
    keys[_KEY_PSEUDONYM] = secret_store.get_or_create()

    token_path = data_dir / "secrets" / "token_vault.key"
    if token_path.exists() or config.privacy.token_vault_enabled:
        token_store = TokenVaultKeyStore(data_dir)
        keys[_KEY_TOKEN_VAULT] = token_store.get_or_create()

    if config.database.encryption_enabled:
        keys[_KEY_SQLCIPHER] = load_sqlcipher_key(config.database, data_dir)

    if config.tracking.encryption_enabled:
        keys[_KEY_TRACKING] = _load_tracking_key(config.tracking, data_dir)

    if config.encryption.enabled:
        keys[_KEY_MEDIA] = _load_media_key(config.encryption)

    return keys


def _store_keys(config: AppConfig, keys: dict[str, bytes]) -> None:
    data_dir = Path(config.capture.data_dir)
    log = get_logger("security.keys")

    if _KEY_PSEUDONYM in keys:
        _write_dpapi_file(data_dir / "secrets" / "pseudonym.key", keys[_KEY_PSEUDONYM])

    if _KEY_TOKEN_VAULT in keys:
        _write_dpapi_file(data_dir / "secrets" / "token_vault.key", keys[_KEY_TOKEN_VAULT])

    if _KEY_SQLCIPHER in keys:
        _store_sqlcipher_key(config.database, data_dir, keys[_KEY_SQLCIPHER])

    if _KEY_TRACKING in keys:
        _store_tracking_key(config.tracking, data_dir, keys[_KEY_TRACKING])

    if _KEY_MEDIA in keys:
        _store_media_key(config.encryption, keys[_KEY_MEDIA])

    log.info("Imported {} keys.", len(keys))


def _derive_key(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_KDF_N,
        r=_KDF_R,
        p=_KDF_P,
        dklen=_KDF_LEN,
        maxmem=_KDF_MAXMEM,
    )


def _load_media_key(config: EncryptionConfig) -> bytes:
    provider = config.key_provider
    if provider == "windows-credential-manager":
        key = _read_windows_credential(config.key_name)
        if key is None:
            key = os.urandom(32)
            _write_windows_credential(config.key_name, key)
        return key
    if provider.startswith("file:"):
        path = Path(provider.split(":", 1)[1]).expanduser().resolve()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            key = os.urandom(32)
            path.write_bytes(key)
            _ensure_private_permissions(path)
            return key
        data = path.read_bytes()
        _ensure_private_permissions(path)
        return data
    if provider.startswith("env:"):
        value = os.getenv(provider.split(":", 1)[1])
        if not value:
            raise RuntimeError("Encryption key environment variable is missing")
        return bytes.fromhex(value)
    raise ValueError(f"Unsupported encryption key provider: {provider}")


def _store_media_key(config: EncryptionConfig, key: bytes) -> None:
    provider = config.key_provider
    if provider == "windows-credential-manager":
        _write_windows_credential(config.key_name, key)
        return
    if provider.startswith("file:"):
        path = Path(provider.split(":", 1)[1]).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
        _ensure_private_permissions(path)
        return
    if provider.startswith("env:"):
        raise RuntimeError("Encryption key provider is env; set the environment variable manually")
    raise ValueError(f"Unsupported encryption key provider: {provider}")


def _load_tracking_key(config: TrackingConfig, data_dir: Path) -> bytes:
    provider = config.encryption_key_provider
    if provider == "env":
        value = os.getenv(config.encryption_env_var)
        if not value:
            raise RuntimeError("Tracking encryption env var missing")
        return bytes.fromhex(value)
    if provider not in {"file", "dpapi_file"}:
        raise ValueError(f"Unsupported tracking encryption provider: {provider}")
    db_path = config.db_path
    if not db_path.is_absolute():
        db_path = data_dir / db_path
    key_path = config.encryption_key_path
    if not key_path.is_absolute():
        key_path = db_path.parent / key_path
    if not key_path.exists():
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key = os.urandom(32)
        _write_dpapi_file(key_path, key, use_dpapi=provider == "dpapi_file")
        return key
    return _read_dpapi_file(key_path, provider == "dpapi_file")


def _store_tracking_key(config: TrackingConfig, data_dir: Path, key: bytes) -> None:
    provider = config.encryption_key_provider
    if provider == "env":
        raise RuntimeError("Tracking key provider is env; set the environment variable manually")
    if provider not in {"file", "dpapi_file"}:
        raise ValueError(f"Unsupported tracking encryption provider: {provider}")
    db_path = config.db_path
    if not db_path.is_absolute():
        db_path = data_dir / db_path
    key_path = config.encryption_key_path
    if not key_path.is_absolute():
        key_path = db_path.parent / key_path
    _write_dpapi_file(key_path, key, use_dpapi=provider == "dpapi_file")


def _store_sqlcipher_key(config: DatabaseConfig, data_dir: Path, key: bytes) -> None:
    provider = config.encryption_provider
    if provider == "env":
        raise RuntimeError("SQLCipher key provider is env; set the environment variable manually")
    if provider not in {"file", "dpapi_file"}:
        raise ValueError(f"Unsupported SQLCipher key provider: {provider}")
    key_path = Path(config.encryption_key_path)
    if not key_path.is_absolute():
        key_path = data_dir / key_path
    _write_dpapi_file(key_path, key, use_dpapi=provider == "dpapi_file")


def _read_dpapi_file(path: Path, use_dpapi: bool = True) -> bytes:
    data = path.read_bytes()
    if not use_dpapi:
        return data
    try:
        import win32crypt  # pragma: no cover - Windows specific

        return win32crypt.CryptUnprotectData(data, None, None, None, 0)[1]
    except Exception:
        return data


def _write_dpapi_file(path: Path, key: bytes, use_dpapi: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if use_dpapi:
        try:
            import win32crypt  # pragma: no cover - Windows specific

            protected = win32crypt.CryptProtectData(key, None, None, None, None, 0)
            path.write_bytes(protected)
            _ensure_private_permissions(path)
            return
        except Exception:
            pass
    path.write_bytes(key)
    _ensure_private_permissions(path)


def _ensure_private_permissions(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o600)
    except Exception:
        return


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _b64decode(text: Any) -> bytes:
    if not isinstance(text, str):
        raise ValueError("Invalid base64 payload")
    return base64.b64decode(text.encode("utf-8"))


def _require_password(password: str) -> None:
    if not password:
        raise RuntimeError("Password is required for key export/import")
