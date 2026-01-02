"""Entity resolution and stable pseudonymization utilities."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sqlalchemy import select

from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import EntityAliasRecord, EntityRecord, EventRecord

EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
DOMAIN_RE = re.compile(r"\b[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE)
WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s\"']+")


@dataclass(frozen=True)
class EntityToken:
    token: str
    entity_type: str
    notes: str | None = None


class SecretStore:
    """Load or create the local secret key for stable pseudonyms."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "secrets" / "pseudonym.key"
        self._log = get_logger("privacy.secret")

    def get_or_create(self) -> bytes:
        if self._path.exists():
            return self._read_key()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        key = os.urandom(32)
        self._write_key(key)
        return key

    def _read_key(self) -> bytes:
        data = self._path.read_bytes()
        if not data:
            raise RuntimeError("Pseudonym key file is empty")
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
            return
        except Exception:
            self._log.warning("DPAPI unavailable; storing pseudonym key on disk.")
        self._path.write_bytes(key)


def stable_token(prefix: str, value: str, secret: bytes) -> str:
    digest = hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{prefix}{digest[:4].upper()}"


class EntityResolver:
    def __init__(self, db: DatabaseManager, secret: bytes) -> None:
        self._db = db
        self._secret = secret
        self._log = get_logger("privacy.entities")

    def resolve_alias(
        self,
        alias_text: str,
        entity_type: str,
        alias_type: str,
        confidence: float,
    ) -> EntityToken:
        alias_norm = alias_text.strip().casefold()
        with self._db.session() as session:
            existing = session.execute(
                select(EntityAliasRecord, EntityRecord)
                .join(EntityRecord, EntityAliasRecord.entity_id == EntityRecord.entity_id)
                .where(EntityAliasRecord.alias_norm == alias_norm)
                .where(EntityRecord.entity_type == entity_type)
            ).all()
            if len(existing) == 1 and existing[0][0].confidence >= 0.85:
                record = existing[0][1]
                return EntityToken(record.canonical_token, record.entity_type)
            if len(existing) > 1:
                token = stable_token(f"{entity_type}_ALIAS_", alias_norm, self._secret)
                entity = EntityRecord(
                    entity_type=entity_type,
                    canonical_name=alias_text,
                    canonical_token=token,
                )
                session.add(entity)
                session.flush()
                session.add(
                    EntityAliasRecord(
                        entity_id=entity.entity_id,
                        alias_text=alias_text,
                        alias_norm=alias_norm,
                        alias_type="ambiguous",
                        confidence=0.5,
                    )
                )
                return EntityToken(token, entity_type, notes="ambiguous alias")
            token = stable_token(f"{entity_type}_", alias_norm, self._secret)
            entity = EntityRecord(
                entity_type=entity_type,
                canonical_name=alias_text,
                canonical_token=token,
            )
            session.add(entity)
            session.flush()
            session.add(
                EntityAliasRecord(
                    entity_id=entity.entity_id,
                    alias_text=alias_text,
                    alias_norm=alias_norm,
                    alias_type=alias_type,
                    confidence=confidence,
                )
            )
            return EntityToken(token, entity_type)

    def tokens_for_events(self, events: Iterable[EventRecord]) -> list[EntityToken]:
        tokens: list[EntityToken] = []
        seen: set[str] = set()
        for event in events:
            if event.app_name and event.app_name not in seen:
                token = self.resolve_alias(event.app_name, "APP", "exact", 0.9)
                if token.token not in seen:
                    tokens.append(token)
                    seen.add(token.token)
            if event.domain and event.domain not in seen:
                token = self.resolve_alias(event.domain, "DOMAIN", "domain", 0.95)
                if token.token not in seen:
                    tokens.append(token)
                    seen.add(token.token)
        return tokens

    def pseudonymize_text(self, text: str) -> str:
        redacted = EMAIL_RE.sub(
            lambda match: stable_token("EMAIL_", match.group(0).casefold(), self._secret),
            text,
        )
        redacted = WINDOWS_PATH_RE.sub(
            lambda match: stable_token("PATH_", match.group(0).casefold(), self._secret),
            redacted,
        )
        redacted = DOMAIN_RE.sub(
            lambda match: stable_token("DOMAIN_", match.group(0).casefold(), self._secret),
            redacted,
        )
        return redacted
