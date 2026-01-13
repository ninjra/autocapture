from __future__ import annotations

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.security.token_vault import TokenVaultStore
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import TokenVaultRecord


def test_token_vault_encrypts_and_resolves() -> None:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    config.privacy.token_vault_enabled = True
    config.privacy.allow_token_vault_decrypt = True
    db = DatabaseManager(config.database)
    vault = TokenVaultStore(config, db)
    vault.record_token("EMAIL_ABC", "EMAIL", "person@example.com")
    with db.session() as session:
        row = session.get(TokenVaultRecord, "EMAIL_ABC")
        assert row is not None
        assert "person@example.com" not in row.value_ciphertext
    resolved = vault.resolve_tokens(["EMAIL_ABC"])
    assert resolved["EMAIL_ABC"] == "person@example.com"


def test_token_vault_respects_decrypt_flag() -> None:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    config.privacy.token_vault_enabled = True
    config.privacy.allow_token_vault_decrypt = False
    db = DatabaseManager(config.database)
    vault = TokenVaultStore(config, db)
    vault.record_token("DOMAIN_ABC", "DOMAIN", "example.com")
    resolved = vault.resolve_tokens(["DOMAIN_ABC"])
    assert resolved == {}
