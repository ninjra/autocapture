"""Encrypted token vault for reversible pseudonymization."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import select

from ..config import AppConfig
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import TokenVaultRecord


def _ensure_private_permissions(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o600)
    except Exception:
        return


class TokenVaultKeyStore:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "secrets" / "token_vault.key"
        self._log = get_logger("privacy.token_vault")

    def get_or_create(self) -> bytes:
        if self._path.exists():
            key = self._read_key()
            _ensure_private_permissions(self._path)
            return key
        self._path.parent.mkdir(parents=True, exist_ok=True)
        key = os.urandom(32)
        self._write_key(key)
        return key

    def _read_key(self) -> bytes:
        data = self._path.read_bytes()
        if not data:
            raise RuntimeError("Token vault key file is empty")
        try:
            import win32crypt  # pragma: no cover - Windows specific

            return win32crypt.CryptUnprotectData(data, None, None, None, 0)[1]
        except Exception:
            return data

    def _write_key(self, key: bytes) -> None:
        try:
            import win32crypt  # pragma: no cover - Windows specific

            protected = win32crypt.CryptProtectData(key, None, None, None, None, 0)
            self._path.write_bytes(protected)
            _ensure_private_permissions(self._path)
            return
        except Exception:
            self._log.warning("DPAPI unavailable; storing token vault key on disk.")
        self._path.write_bytes(key)
        _ensure_private_permissions(self._path)


class TokenVaultStore:
    def __init__(self, config: AppConfig, db: DatabaseManager) -> None:
        self._config = config
        self._db = db
        self._log = get_logger("privacy.token_vault")
        self._key = TokenVaultKeyStore(Path(config.capture.data_dir)).get_or_create()
        self._aesgcm = AESGCM(self._key)

    def record_token(self, token: str, entity_type: str, value: str) -> None:
        if not self._config.privacy.token_vault_enabled:
            return
        value_hash = hashlib.sha256(value.encode("utf-8")).hexdigest()
        ciphertext = self._encrypt(value)

        def _upsert(session) -> None:
            existing = session.execute(
                select(TokenVaultRecord).where(TokenVaultRecord.token == token)
            ).scalar_one_or_none()
            if existing:
                existing.last_seen = _now()
                if existing.value_hash != value_hash:
                    existing.value_hash = value_hash
                    existing.value_ciphertext = ciphertext
                session.add(existing)
            else:
                session.add(
                    TokenVaultRecord(
                        token=token,
                        entity_type=entity_type,
                        value_ciphertext=ciphertext,
                        value_hash=value_hash,
                        created_at=_now(),
                        last_seen=_now(),
                    )
                )

        self._db.transaction(_upsert)

    def resolve_tokens(self, tokens: list[str]) -> dict[str, str]:
        if not self._config.privacy.allow_token_vault_decrypt:
            return {}
        if not tokens:
            return {}
        with self._db.session() as session:
            rows = (
                session.execute(select(TokenVaultRecord).where(TokenVaultRecord.token.in_(tokens)))
                .scalars()
                .all()
            )
        resolved: dict[str, str] = {}
        for row in rows:
            try:
                value = self._decrypt(row.value_ciphertext)
                if hashlib.sha256(value.encode("utf-8")).hexdigest() == row.value_hash:
                    resolved[row.token] = value
            except Exception:
                continue
        return resolved

    def _encrypt(self, plaintext: str) -> str:
        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.b64encode(nonce + ciphertext).decode("utf-8")

    def _decrypt(self, payload: str) -> str:
        raw = base64.b64decode(payload.encode("utf-8"))
        nonce, ciphertext = raw[:12], raw[12:]
        data = self._aesgcm.decrypt(nonce, ciphertext, None)
        return data.decode("utf-8")


def _now():
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc)
