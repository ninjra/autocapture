from pathlib import Path

import pytest
from sqlalchemy import text

from autocapture.config import AppConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.ledger import LedgerWriter


def test_ledger_chain_append_only(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    config = AppConfig()
    config.database.url = f"sqlite:///{db_path.as_posix()}"
    config.database.encryption_enabled = False
    config.database.allow_insecure_dev = True
    db = DatabaseManager(config.database)
    ledger = LedgerWriter(db)

    entry1 = ledger.append_entry("capture", {"frame_id": "f1"})
    entry2 = ledger.append_entry("extract", {"artifact_id": "a1"})

    assert entry1.entry_hash
    assert entry2.entry_hash
    assert ledger.validate_chain(None)

    with pytest.raises(Exception):
        with db.engine.begin() as conn:
            conn.execute(
                text("UPDATE provenance_ledger_entries SET entry_type = 'x' WHERE entry_id = :id"),
                {"id": entry1.entry_id},
            )
