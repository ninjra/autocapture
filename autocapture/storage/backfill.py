"""Resumable backfill for Phase 0 schema upgrades."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sqlalchemy import and_, or_, select

from ..config import AppConfig
from ..image_utils import hash_rgb_image
from ..logging_utils import get_logger
from ..media.store import MediaStore
from ..text.normalize import normalize_text
from .database import DatabaseManager
from .models import (
    BackfillCheckpointRecord,
    CaptureRecord,
    EmbeddingRecord,
    EventRecord,
    OCRSpanRecord,
)


@dataclass(frozen=True)
class BackfillCounts:
    captures_updated: int = 0
    events_updated: int = 0
    spans_updated: int = 0
    embeddings_updated: int = 0
    hashes_computed: int = 0
    normalized_texts: int = 0


class BackfillRunner:
    def __init__(self, config: AppConfig, db: DatabaseManager | None = None) -> None:
        self._config = config
        self._db = db or DatabaseManager(config.database)
        self._log = get_logger("backfill")
        self._media = MediaStore(config.capture, config.encryption)

    def run(
        self,
        *,
        tasks: Iterable[str] | None = None,
        dry_run: bool = False,
        batch_size: int = 500,
        max_rows: int | None = None,
        frame_hash_days: int = 7,
        fill_monotonic: bool = False,
        reset_checkpoints: bool = False,
    ) -> BackfillCounts:
        allowed = {
            "captures",
            "events",
            "spans",
            "embeddings",
        }
        requested = {task.strip().lower() for task in tasks or allowed}
        selected = [task for task in allowed if task in requested]
        total = BackfillCounts()
        for task in selected:
            if task == "captures":
                total = _merge_counts(
                    total,
                    self._backfill_captures(
                        dry_run=dry_run,
                        batch_size=batch_size,
                        max_rows=max_rows,
                        frame_hash_days=frame_hash_days,
                        fill_monotonic=fill_monotonic,
                        reset_checkpoint=reset_checkpoints,
                    ),
                )
            elif task == "events":
                total = _merge_counts(
                    total,
                    self._backfill_events(
                        dry_run=dry_run,
                        batch_size=batch_size,
                        max_rows=max_rows,
                        reset_checkpoint=reset_checkpoints,
                    ),
                )
            elif task == "spans":
                total = _merge_counts(
                    total,
                    self._backfill_spans(
                        dry_run=dry_run,
                        batch_size=batch_size,
                        max_rows=max_rows,
                        reset_checkpoint=reset_checkpoints,
                    ),
                )
            elif task == "embeddings":
                total = _merge_counts(
                    total,
                    self._backfill_embeddings(
                        dry_run=dry_run,
                        batch_size=batch_size,
                        max_rows=max_rows,
                        reset_checkpoint=reset_checkpoints,
                    ),
                )
        return total

    def _backfill_captures(
        self,
        *,
        dry_run: bool,
        batch_size: int,
        max_rows: int | None,
        frame_hash_days: int,
        fill_monotonic: bool,
        reset_checkpoint: bool,
    ) -> BackfillCounts:
        name = "captures"
        processed = 0
        updated = 0
        hashes = 0
        if reset_checkpoint:
            self._clear_checkpoint(name)
        checkpoint = self._load_checkpoint(name)
        last_ts, last_id = _checkpoint_cursor(checkpoint)
        cutoff = None
        if frame_hash_days and frame_hash_days > 0:
            cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=frame_hash_days)

        while True:
            if max_rows is not None and processed >= max_rows:
                break
            with self._db.session() as session:
                stmt = select(CaptureRecord).order_by(
                    CaptureRecord.captured_at.asc(), CaptureRecord.id.asc()
                )
                if last_ts is not None and last_id is not None:
                    stmt = stmt.where(
                        or_(
                            CaptureRecord.captured_at > last_ts,
                            and_(
                                CaptureRecord.captured_at == last_ts,
                                CaptureRecord.id > last_id,
                            ),
                        )
                    )
                batch = session.execute(stmt.limit(batch_size)).scalars().all()
            if not batch:
                break

            def _apply(session) -> None:
                nonlocal updated, hashes, processed, last_ts, last_id
                for record in batch:
                    record = session.merge(record)
                    processed += 1
                    changes = False
                    if not record.event_id:
                        record.event_id = record.id
                        changes = True
                    if not record.created_at_utc and record.captured_at:
                        record.created_at_utc = record.captured_at
                        changes = True
                    if fill_monotonic and record.monotonic_ts is None:
                        # Synthetic monotonic ordering derived from wall-clock capture time.
                        record.monotonic_ts = record.captured_at.timestamp()
                        changes = True
                    if not record.privacy_flags:
                        record.privacy_flags = {
                            "excluded": False,
                            "masked_regions_applied": False,
                            "cloud_allowed": False,
                        }
                        changes = True
                    if not record.schema_version:
                        record.schema_version = "v1"
                        changes = True
                    if (
                        self._config.features.enable_frame_hash
                        and not record.frame_hash
                        and record.image_path
                        and (cutoff is None or record.captured_at >= cutoff)
                    ):
                        try:
                            path = Path(record.image_path)
                            image = self._media.read_image(path)
                            record.frame_hash = hash_rgb_image(image)
                            hashes += 1
                            changes = True
                        except Exception:
                            pass
                    if changes:
                        updated += 1
                    last_ts = record.captured_at
                    last_id = record.id

            if not dry_run:
                self._db.transaction(_apply)
            else:
                # Dry-run still advances the cursor without persisting.
                for record in batch:
                    processed += 1
                    last_ts = record.captured_at
                    last_id = record.id
            self._save_checkpoint(name, last_ts, last_id)
        if updated:
            self._log.info("Backfilled capture records: {}", updated)
        return BackfillCounts(captures_updated=updated, hashes_computed=hashes)

    def _backfill_events(
        self,
        *,
        dry_run: bool,
        batch_size: int,
        max_rows: int | None,
        reset_checkpoint: bool,
    ) -> BackfillCounts:
        name = "events"
        processed = 0
        updated = 0
        normalized = 0
        if reset_checkpoint:
            self._clear_checkpoint(name)
        checkpoint = self._load_checkpoint(name)
        last_ts, last_id = _checkpoint_cursor(checkpoint)

        while True:
            if max_rows is not None and processed >= max_rows:
                break
            with self._db.session() as session:
                stmt = select(EventRecord).order_by(
                    EventRecord.ts_start.asc(), EventRecord.event_id.asc()
                )
                if last_ts is not None and last_id is not None:
                    stmt = stmt.where(
                        or_(
                            EventRecord.ts_start > last_ts,
                            and_(
                                EventRecord.ts_start == last_ts,
                                EventRecord.event_id > last_id,
                            ),
                        )
                    )
                batch = session.execute(stmt.limit(batch_size)).scalars().all()
            if not batch:
                break

            def _apply(session) -> None:
                nonlocal updated, normalized, processed, last_ts, last_id
                for record in batch:
                    record = session.merge(record)
                    processed += 1
                    changes = False
                    if record.ocr_text and not record.ocr_text_normalized:
                        if self._config.features.enable_normalized_indexing:
                            record.ocr_text_normalized = normalize_text(record.ocr_text)
                            normalized += 1
                            changes = True
                    if not record.frame_hash:
                        capture = session.get(CaptureRecord, record.event_id)
                        if capture and capture.frame_hash:
                            record.frame_hash = capture.frame_hash
                            changes = True
                    if changes:
                        updated += 1
                    last_ts = record.ts_start
                    last_id = record.event_id

            if not dry_run:
                self._db.transaction(_apply)
            else:
                for record in batch:
                    processed += 1
                    last_ts = record.ts_start
                    last_id = record.event_id
            self._save_checkpoint(name, last_ts, last_id)
        if updated:
            self._log.info("Backfilled event records: {}", updated)
        return BackfillCounts(events_updated=updated, normalized_texts=normalized)

    def _backfill_spans(
        self,
        *,
        dry_run: bool,
        batch_size: int,
        max_rows: int | None,
        reset_checkpoint: bool,
    ) -> BackfillCounts:
        name = "spans"
        processed = 0
        updated = 0
        if reset_checkpoint:
            self._clear_checkpoint(name)
        checkpoint = self._load_checkpoint(name)
        last_id = checkpoint.get("last_id") if checkpoint else None

        while True:
            if max_rows is not None and processed >= max_rows:
                break
            with self._db.session() as session:
                stmt = select(OCRSpanRecord).order_by(OCRSpanRecord.id.asc())
                if last_id is not None:
                    stmt = stmt.where(OCRSpanRecord.id > int(last_id))
                batch = session.execute(stmt.limit(batch_size)).scalars().all()
            if not batch:
                break

            def _apply(session) -> None:
                nonlocal updated, processed, last_id
                for record in batch:
                    record = session.merge(record)
                    processed += 1
                    changes = False
                    if not record.schema_version:
                        record.schema_version = "v1"
                        changes = True
                    if not record.engine:
                        record.engine = "unknown"
                        changes = True
                    if not record.frame_hash:
                        capture = session.get(CaptureRecord, record.capture_id)
                        if capture and capture.frame_hash:
                            record.frame_hash = capture.frame_hash
                            changes = True
                    if changes:
                        updated += 1
                    last_id = record.id

            if not dry_run:
                self._db.transaction(_apply)
            else:
                for record in batch:
                    processed += 1
                    last_id = record.id
            self._save_checkpoint(name, None, None, extra={"last_id": last_id})
        if updated:
            self._log.info("Backfilled OCR spans: {}", updated)
        return BackfillCounts(spans_updated=updated)

    def _backfill_embeddings(
        self,
        *,
        dry_run: bool,
        batch_size: int,
        max_rows: int | None,
        reset_checkpoint: bool,
    ) -> BackfillCounts:
        name = "embeddings"
        processed = 0
        updated = 0
        if reset_checkpoint:
            self._clear_checkpoint(name)
        checkpoint = self._load_checkpoint(name)
        last_id = checkpoint.get("last_id") if checkpoint else None

        while True:
            if max_rows is not None and processed >= max_rows:
                break
            with self._db.session() as session:
                stmt = select(EmbeddingRecord).order_by(EmbeddingRecord.id.asc())
                if last_id is not None:
                    stmt = stmt.where(EmbeddingRecord.id > int(last_id))
                batch = session.execute(stmt.limit(batch_size)).scalars().all()
            if not batch:
                break

            def _apply(session) -> None:
                nonlocal updated, processed, last_id
                for record in batch:
                    record = session.merge(record)
                    processed += 1
                    changes = False
                    if not record.frame_hash:
                        capture = session.get(CaptureRecord, record.capture_id)
                        if capture and capture.frame_hash:
                            record.frame_hash = capture.frame_hash
                            changes = True
                    if changes:
                        updated += 1
                    last_id = record.id

            if not dry_run:
                self._db.transaction(_apply)
            else:
                for record in batch:
                    processed += 1
                    last_id = record.id
            self._save_checkpoint(name, None, None, extra={"last_id": last_id})
        if updated:
            self._log.info("Backfilled embeddings: {}", updated)
        return BackfillCounts(embeddings_updated=updated)

    def _load_checkpoint(self, name: str) -> dict:
        with self._db.session() as session:
            record = session.get(BackfillCheckpointRecord, name)
            return dict(record.payload_json or {}) if record else {}

    def _clear_checkpoint(self, name: str) -> None:
        def _delete(session) -> None:
            record = session.get(BackfillCheckpointRecord, name)
            if record:
                session.delete(record)

        self._db.transaction(_delete)

    def _save_checkpoint(
        self, name: str, last_ts: dt.datetime | None, last_id: str | None, *, extra: dict | None = None
    ) -> None:
        payload: dict = dict(extra or {})
        if last_ts is not None and last_id is not None:
            payload.update({"last_ts": last_ts.isoformat(), "last_id": last_id})

        def _write(session) -> None:
            record = session.get(BackfillCheckpointRecord, name)
            if record:
                record.payload_json = payload
                record.updated_at = dt.datetime.now(dt.timezone.utc)
            else:
                session.add(
                    BackfillCheckpointRecord(
                        name=name,
                        updated_at=dt.datetime.now(dt.timezone.utc),
                        payload_json=payload,
                    )
                )

        self._db.transaction(_write)


def _checkpoint_cursor(checkpoint: dict) -> tuple[dt.datetime | None, str | None]:
    last_ts = checkpoint.get("last_ts")
    last_id = checkpoint.get("last_id")
    if isinstance(last_ts, str):
        try:
            last_ts = dt.datetime.fromisoformat(last_ts)
        except Exception:
            last_ts = None
    if not isinstance(last_ts, dt.datetime):
        last_ts = None
    if not isinstance(last_id, str):
        last_id = None
    return last_ts, last_id


def _merge_counts(base: BackfillCounts, incoming: BackfillCounts) -> BackfillCounts:
    return BackfillCounts(
        captures_updated=base.captures_updated + incoming.captures_updated,
        events_updated=base.events_updated + incoming.events_updated,
        spans_updated=base.spans_updated + incoming.spans_updated,
        embeddings_updated=base.embeddings_updated + incoming.embeddings_updated,
        hashes_computed=base.hashes_computed + incoming.hashes_computed,
        normalized_texts=base.normalized_texts + incoming.normalized_texts,
    )
