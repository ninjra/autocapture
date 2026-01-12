"""Entity resolution and stable pseudonymization utilities."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import string
import unicodedata
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


def _ensure_private_permissions(path: Path) -> None:
    """Best-effort tighten permissions for secrets on POSIX.

    Windows file ACLs differ; additionally, when DPAPI is available we store
    encrypted bytes anyway.
    """

    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o600)
    except Exception:
        # Not fatal (e.g., running on a filesystem that doesn't support chmod).
        return


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
            _ensure_private_permissions(self._path)
            return
        except Exception:
            self._log.warning("DPAPI unavailable; storing pseudonym key on disk.")
        self._path.write_bytes(key)
        _ensure_private_permissions(self._path)


def stable_token(prefix: str, value: str, secret: bytes, length: int = 20) -> str:
    digest = hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{prefix}{digest[:length].upper()}"


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
        alias_norm = normalize_alias(alias_text)
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
                entity = self._insert_entity_with_token(session, entity_type, alias_text, token)
                session.add(
                    EntityAliasRecord(
                        entity_id=entity.entity_id,
                        alias_text=alias_text,
                        alias_norm=alias_norm,
                        alias_type="ambiguous",
                        confidence=0.5,
                    )
                )
                return EntityToken(entity.canonical_token, entity_type, notes="ambiguous alias")
            entity = self._insert_entity(session, entity_type, alias_norm, alias_text)
            session.add(
                EntityAliasRecord(
                    entity_id=entity.entity_id,
                    alias_text=alias_text,
                    alias_norm=alias_norm,
                    alias_type=alias_type,
                    confidence=confidence,
                )
            )
            return EntityToken(entity.canonical_token, entity_type)

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

    def pseudonymize_text_with_mapping(
        self, text: str
    ) -> tuple[str, list[tuple[int, int, int, int]]]:
        replacements = _collect_replacements(text, self._secret)
        if not replacements:
            return text, []
        parts: list[str] = []
        mapping: list[tuple[int, int, int, int]] = []
        cursor = 0
        new_index = 0
        for start, end, token in replacements:
            if start < cursor:
                continue
            parts.append(text[cursor:start])
            new_index += start - cursor
            parts.append(token)
            mapping.append((start, end, new_index, new_index + len(token)))
            new_index += len(token)
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts), mapping

    def _insert_entity(
        self,
        session,
        entity_type: str,
        alias_norm: str,
        alias_text: str,
    ) -> EntityRecord:
        base_token = stable_token(f"{entity_type}_", alias_norm, self._secret)
        return self._insert_entity_with_token(session, entity_type, alias_text, base_token)

    def _insert_entity_with_token(
        self,
        session,
        entity_type: str,
        alias_text: str,
        base_token: str,
    ) -> EntityRecord:
        alias_norm = normalize_alias(alias_text)
        for attempt in range(5):
            token = base_token if attempt == 0 else f"{base_token}-{attempt}"
            existing = (
                session.execute(select(EntityRecord).where(EntityRecord.canonical_token == token))
                .scalars()
                .first()
            )
            if existing:
                existing_norm = normalize_alias(existing.canonical_name)
                if existing_norm == alias_norm:
                    return existing
                if existing.canonical_name != alias_text:
                    continue
                return existing
            entity = EntityRecord(
                entity_type=entity_type,
                canonical_name=alias_text,
                canonical_token=token,
            )
            session.add(entity)
            try:
                session.flush()
                return entity
            except Exception:
                session.rollback()
                continue
        token = f"{base_token}-overflow"
        entity = EntityRecord(
            entity_type=entity_type,
            canonical_name=alias_text,
            canonical_token=token,
        )
        session.add(entity)
        session.flush()
        return entity


def _collect_replacements(text: str, secret: bytes) -> list[tuple[int, int, str]]:
    replacements: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern, prefix in (
        (EMAIL_RE, "EMAIL_"),
        (WINDOWS_PATH_RE, "PATH_"),
        (DOMAIN_RE, "DOMAIN_"),
    ):
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < occ_end and end > occ_start for occ_start, occ_end in occupied):
                continue
            token = stable_token(prefix, match.group(0).casefold(), secret)
            replacements.append((start, end, token))
            occupied.append((start, end))
    replacements.sort(key=lambda item: item[0])
    return replacements


def normalize_alias(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    leet_map = str.maketrans(
        {
            "0": "o",
            "1": "l",
            "3": "e",
            "4": "a",
            "5": "s",
            "7": "t",
            "@": "a",
            "$": "s",
        }
    )
    normalized = normalized.translate(leet_map)
    normalized = " ".join(normalized.split())
    normalized = normalized.strip(string.punctuation + " ")
    return normalized
