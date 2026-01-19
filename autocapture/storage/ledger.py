"""Append-only provenance ledger utilities."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import select

from ..contracts_utils import hash_canonical, stable_id
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import ProvenanceLedgerEntryRecord


class LedgerWriter:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._log = get_logger("ledger")

    def append_entry(
        self,
        entry_type: str,
        payload: dict[str, Any],
        *,
        answer_id: str | None = None,
        created_at: dt.datetime | None = None,
    ) -> ProvenanceLedgerEntryRecord:
        created_at = created_at or dt.datetime.now(dt.timezone.utc)
        prev_hash = self._latest_hash(answer_id)
        entry_payload = {
            "answer_id": answer_id,
            "entry_type": entry_type,
            "payload": payload,
            "created_at": created_at,
            "prev_hash": prev_hash,
        }
        entry_hash = hash_canonical(entry_payload)
        entry_id = stable_id("ledger", {"entry_hash": entry_hash, "answer_id": answer_id})

        def _write(session):
            existing = session.get(ProvenanceLedgerEntryRecord, entry_id)
            if existing:
                if existing.entry_hash != entry_hash:
                    raise RuntimeError("Ledger entry hash mismatch for idempotent write")
                return existing
            record = ProvenanceLedgerEntryRecord(
                entry_id=entry_id,
                answer_id=answer_id,
                entry_type=entry_type,
                payload_json=payload,
                prev_hash=prev_hash,
                entry_hash=entry_hash,
                schema_version=1,
                created_at=created_at,
            )
            session.add(record)
            return record

        return self._db.transaction(_write)

    def validate_chain(self, answer_id: str | None) -> bool:
        with self._db.session() as session:
            rows = (
                session.execute(
                    select(ProvenanceLedgerEntryRecord)
                    .where(ProvenanceLedgerEntryRecord.answer_id == answer_id)
                    .order_by(ProvenanceLedgerEntryRecord.created_at.asc())
                )
                .scalars()
                .all()
            )
        prev_hash: str | None = None
        for row in rows:
            entry_payload = {
                "answer_id": row.answer_id,
                "entry_type": row.entry_type,
                "payload": row.payload_json,
                "created_at": row.created_at,
                "prev_hash": prev_hash,
            }
            computed = hash_canonical(entry_payload)
            if row.prev_hash != prev_hash or row.entry_hash != computed:
                self._log.warning("Ledger chain invalid at entry {}", row.entry_id)
                return False
            prev_hash = row.entry_hash
        return True

    def _latest_hash(self, answer_id: str | None) -> str | None:
        with self._db.session() as session:
            row = (
                session.execute(
                    select(ProvenanceLedgerEntryRecord.entry_hash)
                    .where(ProvenanceLedgerEntryRecord.answer_id == answer_id)
                    .order_by(ProvenanceLedgerEntryRecord.created_at.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
        return row
