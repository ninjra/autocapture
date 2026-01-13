from __future__ import annotations

import pytest

from autocapture.config import AppConfig, DatabaseConfig


def test_remote_postgres_requires_tls() -> None:
    with pytest.raises(ValueError):
        AppConfig(database=DatabaseConfig(url="postgresql://user:pass@db.example.com:5432/db"))


def test_remote_postgres_with_tls_ok() -> None:
    config = AppConfig(
        database=DatabaseConfig(url="postgresql://user:pass@db.example.com:5432/db?sslmode=require")
    )
    assert "sslmode=require" in config.database.url
