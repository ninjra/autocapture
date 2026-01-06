from autocapture.config import DatabaseConfig
from autocapture.memory.entities import EntityResolver
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EntityAliasRecord, EntityRecord


def test_ambiguous_alias_creates_alias_token() -> None:
    db = DatabaseManager(DatabaseConfig(url="sqlite:///:memory:"))
    secret = b"secret-key"
    resolver = EntityResolver(db, secret)

    with db.session() as session:
        entity_a = EntityRecord(
            entity_type="ORG", canonical_name="Acme A", canonical_token="ORG_AAAA"
        )
        entity_b = EntityRecord(
            entity_type="ORG", canonical_name="Acme B", canonical_token="ORG_BBBB"
        )
        session.add_all([entity_a, entity_b])
        session.flush()
        session.add_all(
            [
                EntityAliasRecord(
                    entity_id=entity_a.entity_id,
                    alias_text="Acme",
                    alias_norm="acme",
                    alias_type="exact",
                    confidence=0.9,
                ),
                EntityAliasRecord(
                    entity_id=entity_b.entity_id,
                    alias_text="Acme",
                    alias_norm="acme",
                    alias_type="exact",
                    confidence=0.9,
                ),
            ]
        )

    token = resolver.resolve_alias("Acme", "ORG", "exact", 0.9)
    assert token.token.startswith("ORG_ALIAS_")


def test_case_insensitive_alias_resolution() -> None:
    db = DatabaseManager(DatabaseConfig(url="sqlite:///:memory:"))
    secret = b"secret-key"
    resolver = EntityResolver(db, secret)

    token_a = resolver.resolve_alias("Acme", "ORG", "exact", 0.9)
    token_b = resolver.resolve_alias("ACME", "ORG", "exact", 0.9)

    assert token_a.token == token_b.token


def test_leetspeak_alias_resolution() -> None:
    db = DatabaseManager(DatabaseConfig(url="sqlite:///:memory:"))
    secret = b"secret-key"
    resolver = EntityResolver(db, secret)

    token_a = resolver.resolve_alias("Microsoft", "ORG", "exact", 0.9)
    token_b = resolver.resolve_alias("Micros0ft", "ORG", "exact", 0.9)

    assert token_a.token == token_b.token
