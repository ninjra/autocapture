"""Environment-backed secret store abstraction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .redaction import redact_value


@dataclass(frozen=True)
class SecretRecord:
    key: str
    value: str
    source: str


class SecretStore:
    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._env = env or os.environ

    def get(self, key: str) -> SecretRecord | None:
        value = self._env.get(key)
        if value is None:
            return None
        return SecretRecord(key=key, value=value, source="env")

    def require(self, key: str) -> SecretRecord:
        record = self.get(key)
        if record is None:
            raise RuntimeError(f"Missing required secret: {key}")
        return record

    @staticmethod
    def redact(record: SecretRecord | None) -> str | None:
        if record is None:
            return None
        return str(redact_value(record.value))
