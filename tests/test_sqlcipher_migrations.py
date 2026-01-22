import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import text

from autocapture.config import DatabaseConfig
from autocapture.storage.database import DatabaseManager


@pytest.mark.skipif(
    importlib.util.find_spec("pysqlcipher3") is None,
    reason="SQLCipher driver not installed",
)
def test_sqlcipher_migrations_use_encrypted_engine(tmp_path: Path) -> None:
    db_path = tmp_path / "autocapture.db"
    config = DatabaseConfig(
        url=f"sqlite:///{db_path.as_posix()}",
        encryption_enabled=True,
        encryption_provider="file",
        encryption_key_path=Path("secrets/sqlcipher.key"),
    )
    db = DatabaseManager(config)
    with db.session() as session:
        session.execute(text("SELECT 1"))
