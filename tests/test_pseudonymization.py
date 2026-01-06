from pathlib import Path

import pytest

from autocapture.memory.entities import SecretStore, stable_token
from autocapture.memory.entities import EntityResolver
from autocapture.config import DatabaseConfig
from autocapture.storage.database import DatabaseManager


def test_stable_token_is_deterministic(tmp_path: Path) -> None:
    store = SecretStore(tmp_path)
    key = store.get_or_create()
    token_a = stable_token("ORG_", "example", key)
    token_b = stable_token("ORG_", "example", key)
    assert token_a == token_b
    assert token_a.startswith("ORG_")
    assert len(token_a) == len("ORG_") + 20
    assert token_a[len("ORG_") :].isalnum()


def test_stable_token_collision_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    db = DatabaseManager(DatabaseConfig(url="sqlite:///:memory:"))
    resolver = EntityResolver(db, b"secret")

    def fake_token(prefix: str, value: str, secret: bytes, length: int = 20) -> str:
        return f"{prefix}COLLIDE"

    monkeypatch.setattr("autocapture.memory.entities.stable_token", fake_token)

    token_a = resolver.resolve_alias("alpha", "ORG", "exact", 0.9)
    token_b = resolver.resolve_alias("beta", "ORG", "exact", 0.9)

    assert token_a.token != token_b.token
