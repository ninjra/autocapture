from __future__ import annotations

import datetime as dt
from pathlib import Path

from autocapture.answer.integrity import check_citations
from autocapture.config import AppConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.ledger import LedgerWriter
from autocapture.storage.models import ArtifactRecord, CitableSpanRecord, FrameRecord


def test_integrity_checker_missing_media(tmp_path: Path) -> None:
    db_path = tmp_path / "integrity.db"
    media_path = tmp_path / "frame.webp"
    media_path.write_bytes(b"test")

    config = AppConfig()
    config.database.url = f"sqlite:///{db_path.as_posix()}"
    config.database.encryption_enabled = False
    config.database.allow_insecure_dev = True
    db = DatabaseManager(config.database)

    now = dt.datetime.now(dt.timezone.utc)
    span_id = "span_1"
    with db.session() as session:
        session.add(
            FrameRecord(
                frame_id="f1",
                event_id=None,
                captured_at_utc=now,
                monotonic_ts=0.0,
                monitor_id="m1",
                monitor_bounds=[0, 0, 100, 100],
                app_name="app",
                window_title="title",
                media_path=str(media_path),
                privacy_flags={},
                frame_hash="h",
                excluded=False,
                masked=False,
                schema_version=1,
                created_at=now,
            )
        )
        session.flush()
        session.add(
            ArtifactRecord(
                artifact_id="a1",
                frame_id="f1",
                event_id=None,
                artifact_type="ocr",
                engine="test",
                engine_version="v1",
                derived_from={"frame_hash": "h"},
                upstream_artifact_ids=[],
                schema_version=1,
                created_at=now,
            )
        )
        session.add(
            CitableSpanRecord(
                span_id=span_id,
                artifact_id="a1",
                frame_id="f1",
                event_id=None,
                span_hash="s",
                text="text",
                start_offset=0,
                end_offset=4,
                bbox=[0, 0, 10, 10],
                bbox_norm=None,
                tombstoned=False,
                expires_at_utc=None,
                legacy_span_key="S1",
                schema_version=1,
                created_at=now,
            )
        )
    ledger = LedgerWriter(db)
    ledger.append_entry("span", {"span_id": span_id})

    result = check_citations(db, [span_id])
    assert span_id in result.valid_span_ids

    media_path.unlink()
    result = check_citations(db, [span_id])
    assert span_id in result.invalid_span_ids
