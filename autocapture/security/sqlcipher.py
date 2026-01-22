"""SQLCipher key management utilities."""

from __future__ import annotations

import os
from pathlib import Path

from ..config import DatabaseConfig
from ..logging_utils import get_logger


def _ensure_private_permissions(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o600)
    except Exception:
        return


def load_sqlcipher_key(config: DatabaseConfig, data_dir: Path) -> bytes:
    provider = config.encryption_provider
    log = get_logger("db.sqlcipher")
    if provider == "env":
        value = os.getenv(config.encryption_env_var)
        if not value:
            raise RuntimeError("SQLCipher key env var is missing")
        return bytes.fromhex(value)
    if provider in {"file", "dpapi_file"}:
        path = Path(config.encryption_key_path)
        if not path.is_absolute():
            path = data_dir / path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            key = os.urandom(32)
            _write_key(path, key, provider == "dpapi_file", log)
            _ensure_private_permissions(path)
            return key
        data = path.read_bytes()
        if provider == "dpapi_file":
            try:
                import win32crypt  # pragma: no cover - Windows specific

                data = win32crypt.CryptUnprotectData(data, None, None, None, 0)[1]
            except Exception:
                log.warning("DPAPI unavailable; using raw key file")
        _ensure_private_permissions(path)
        return data
    raise ValueError(f"Unsupported SQLCipher key provider: {provider}")


def _write_key(path: Path, key: bytes, use_dpapi: bool, log) -> None:
    if use_dpapi:
        try:
            import win32crypt  # pragma: no cover - Windows specific

            protected = win32crypt.CryptProtectData(key, None, None, None, None, 0)
            path.write_bytes(protected)
            return
        except Exception:
            log.warning("DPAPI unavailable; storing SQLCipher key on disk.")
    path.write_bytes(key)


def ensure_sqlcipher_create_function_compat(sqlcipher_module) -> None:
    """Ensure SQLCipher connections accept deterministic kwarg for create_function."""

    conn_cls = getattr(sqlcipher_module, "Connection", None)
    if conn_cls is None:
        return
    if getattr(conn_cls, "_autocapture_create_function_compat", False):
        return
    original = getattr(conn_cls, "create_function", None)
    if original is None:
        return

    def _compat(self, name, num_params, func, deterministic=None):  # noqa: ANN001
        _ = deterministic
        return original(self, name, num_params, func)

    try:
        setattr(conn_cls, "create_function", _compat)
        setattr(conn_cls, "_autocapture_create_function_compat", True)
    except Exception:
        return
