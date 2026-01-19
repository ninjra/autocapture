"""Synthetic fixtures for Next-10 gates and tests."""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import AppConfig, DatabaseConfig
from ..contracts_utils import hash_canonical, stable_id
from ..image_utils import hash_rgb_image
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.ledger import LedgerWriter
from ..storage.models import (
    ArtifactRecord,
    CaptureRecord,
    CitableSpanRecord,
    EventRecord,
    FrameRecord,
    OCRSpanRecord,
)
from ..vision.types import build_ocr_payload
from ..indexing.lexical_index import LexicalIndex


@dataclass(frozen=True)
class SyntheticCorpus:
    db: DatabaseManager
    config: AppConfig
    data_dir: Path
    event_ids: list[str]
    span_ids: list[str]
    query_map: dict[str, list[str]]


def build_test_config(*, db_path: Path, data_dir: Path, enable_rerank: bool = True) -> AppConfig:
    config = AppConfig(
        database=DatabaseConfig(url=f"sqlite:///{db_path}", sqlite_wal=False),
    )
    config.capture.data_dir = data_dir
    config.offline = True
    config.embed.text_model = "local-test"
    config.qdrant.enabled = False
    config.retrieval.sparse_enabled = False
    config.retrieval.late_enabled = False
    config.retrieval.fusion_enabled = True
    config.routing.reranker = "enabled" if enable_rerank else "disabled"
    config.reranker.enabled = enable_rerank
    config.reranker.model = "local-test"
    config.model_stages.query_refine.enabled = False
    config.model_stages.draft_generate.enabled = False
    config.model_stages.final_answer.enabled = False
    config.model_stages.tool_transform.enabled = False
    config.next10.enabled = True
    config.features.enable_thresholding = True
    return config


def seed_synthetic_corpus(
    db: DatabaseManager, data_dir: Path, *, config: AppConfig | None = None
) -> SyntheticCorpus:
    log = get_logger("next10.fixtures")
    media_dir = data_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime(2026, 1, 17, 12, 0, tzinfo=dt.timezone.utc)
    fixtures = [
        ("EVT-1", "Project Alpha cost $100", now - dt.timedelta(hours=2)),
        ("EVT-2", "Project Alpha cost $150", now - dt.timedelta(hours=1)),
        ("EVT-3", "Beta status: open", now - dt.timedelta(hours=3)),
        ("EVT-4", "Release 2.1 scheduled 2026-02-01", now - dt.timedelta(hours=4)),
        ("EVT-5", "Support ticket ABC-999 closed", now - dt.timedelta(hours=5)),
    ]
    event_ids: list[str] = []
    span_ids: list[str] = []
    query_map: dict[str, list[str]] = {
        "alpha cost": ["EVT-1", "EVT-2"],
        "beta status": ["EVT-3"],
        "release 2.1": ["EVT-4"],
        "ticket closed": ["EVT-5"],
    }

    def _write_image(event_id: str) -> Path:
        rng = np.random.default_rng(abs(hash(event_id)) % (2**32))
        pixels = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
        img = Image.fromarray(pixels, mode="RGB")
        path = media_dir / f"{event_id}.png"
        img.save(path)
        return path

    events: list[EventRecord] = []
    span_ledger_entries: list[dict] = []
    with db.session() as session:
        for event_id, text, timestamp in fixtures:
            image_path = _write_image(event_id)
            frame_hash = hash_rgb_image(np.asarray(Image.open(image_path)))
            ocr_text, ocr_spans = build_ocr_payload([(text, 0.95, [0, 0, 10, 10])])
            capture = CaptureRecord(
                id=event_id,
                event_id=event_id,
                captured_at=timestamp,
                created_at_utc=timestamp,
                monotonic_ts=1.0,
                image_path=str(image_path),
                focus_path=None,
                foreground_process="App",
                foreground_window="Window",
                monitor_id="m1",
                monitor_bounds=[0, 0, 96, 64],
                is_fullscreen=False,
                privacy_flags={},
                frame_hash=frame_hash,
                schema_version="v1",
                ocr_status="done",
            )
            frame = FrameRecord(
                frame_id=event_id,
                event_id=event_id,
                captured_at_utc=timestamp,
                monotonic_ts=1.0,
                monitor_id="m1",
                monitor_bounds=[0, 0, 96, 64],
                app_name="App",
                window_title="Window",
                media_path=str(image_path),
                privacy_flags={},
                frame_hash=frame_hash,
                excluded=False,
                masked=False,
                schema_version=1,
                created_at=timestamp,
            )
            event = EventRecord(
                event_id=event_id,
                ts_start=timestamp,
                ts_end=None,
                app_name="App",
                window_title="Window",
                url=None,
                domain=None,
                screenshot_path=str(image_path),
                screenshot_hash=frame_hash,
                ocr_text=ocr_text,
                ocr_text_normalized=ocr_text.lower(),
                tags={},
            )
            artifact_id = stable_id(
                "artifact",
                {
                    "frame_id": event_id,
                    "artifact_type": "ocr",
                    "engine": "synthetic",
                    "engine_version": "v1",
                },
            )
            session.add(event)
            session.add(frame)
            session.add(capture)
            session.flush()
            session.add(
                ArtifactRecord(
                    artifact_id=artifact_id,
                    frame_id=event_id,
                    event_id=event_id,
                    artifact_type="ocr",
                    engine="synthetic",
                    engine_version="v1",
                    derived_from={"frame_hash": frame_hash},
                    upstream_artifact_ids=[],
                    schema_version=1,
                    created_at=timestamp,
                )
            )
            for ocr_span in ocr_spans:
                span_key = str(ocr_span.get("span_key", "S1"))
                bbox = ocr_span.get("bbox") or [0, 0, 10, 10]
                span_payload = {
                    "text": ocr_span.get("text", ""),
                    "start": int(ocr_span.get("start", 0)),
                    "end": int(ocr_span.get("end", 0)),
                    "bbox": bbox,
                    "frame_hash": frame_hash,
                }
                span_hash = hash_canonical(span_payload)
                span_id = stable_id("span", {"span_hash": span_hash, "frame_id": event_id})
                session.add(
                    OCRSpanRecord(
                        capture_id=event_id,
                        span_key=span_key,
                        start=int(ocr_span.get("start", 0)),
                        end=int(ocr_span.get("end", 0)),
                        text=ocr_span.get("text", ""),
                        confidence=float(ocr_span.get("conf", 0.9)),
                        bbox=bbox,
                        engine="synthetic",
                        frame_hash=frame_hash,
                        schema_version="v1",
                    )
                )
                session.add(
                    CitableSpanRecord(
                        span_id=span_id,
                        artifact_id=artifact_id,
                        frame_id=event_id,
                        event_id=event_id,
                        span_hash=span_hash,
                        text=ocr_span.get("text", ""),
                        start_offset=int(ocr_span.get("start", 0)),
                        end_offset=int(ocr_span.get("end", 0)),
                        bbox=bbox,
                        bbox_norm=[0.0, 0.0, 1.0, 1.0],
                        tombstoned=False,
                        expires_at_utc=None,
                        legacy_span_key=span_key,
                        schema_version=1,
                        created_at=timestamp,
                    )
                )
                span_ids.append(span_id)
                span_ledger_entries.append(
                    {
                        "span_id": span_id,
                        "artifact_id": artifact_id,
                        "frame_id": event_id,
                        "engine": "synthetic",
                    }
                )
            events.append(event)
            event_ids.append(event_id)

    LexicalIndex(db).bulk_upsert(events)
    ledger = LedgerWriter(db)
    for entry in span_ledger_entries:
        ledger.append_entry("span", entry)
    log.info("Seeded synthetic corpus with {} events", len(event_ids))
    return SyntheticCorpus(
        db=db,
        config=config or AppConfig(),
        data_dir=data_dir,
        event_ids=event_ids,
        span_ids=span_ids,
        query_map=query_map,
    )


def create_synthetic_corpus(enable_rerank: bool = True) -> SyntheticCorpus:
    tmp_dir = Path(tempfile.mkdtemp(prefix="next10-corpus-"))
    db_path = tmp_dir / "synthetic.db"
    config = build_test_config(db_path=db_path, data_dir=tmp_dir, enable_rerank=enable_rerank)
    db = DatabaseManager(config.database)
    os.environ.setdefault("AUTOCAPTURE_TEST_MODE", "1")
    corpus = seed_synthetic_corpus(db, tmp_dir, config=config)
    return SyntheticCorpus(
        db=db,
        config=config,
        data_dir=tmp_dir,
        event_ids=corpus.event_ids,
        span_ids=corpus.span_ids,
        query_map=corpus.query_map,
    )
