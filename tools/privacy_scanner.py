"""Privacy regression scanner for excluded/masked invariants."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from autocapture.capture.privacy_filter import apply_exclude_region_masks
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.image_utils import hash_rgb_image
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import (
    ArtifactRecord,
    CaptureRecord,
    CitableSpanRecord,
    EventRecord,
    FrameRecord,
)
from autocapture.worker.event_worker import EventIngestWorker
from autocapture.vision.types import ExtractionResult, VISION_SCHEMA_VERSION, build_ocr_payload


class _StubExtractor:
    def extract(self, image: np.ndarray) -> ExtractionResult:
        text, spans = build_ocr_payload([("MASKED_MARKER_123", 0.95, [0, 0, 10, 10])])
        tags = {
            "vision_extract": {
                "schema_version": VISION_SCHEMA_VERSION,
                "engine": "stub",
                "screen_summary": "",
                "regions": [],
                "visible_text": text,
                "content_flags": [],
                "tables_detected": None,
                "spreadsheets_detected": None,
                "parse_failed": False,
                "parse_format": "stub",
                "tiles": [],
            }
        }
        return ExtractionResult(text=text, spans=spans, tags=tags)


def _write_report(report: dict, *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    os.environ.setdefault("AUTOCAPTURE_TEST_MODE", "1")
    tmp_dir = Path(tempfile.mkdtemp(prefix="privacy-scan-"))
    db_path = tmp_dir / "privacy.db"
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{db_path}", sqlite_wal=False))
    config.capture.data_dir = tmp_dir
    config.offline = True
    config.qdrant.enabled = False
    db = DatabaseManager(config.database)

    excluded_id = "CAP-EX"
    masked_id = "CAP-MASK"
    now = dt.datetime.now(dt.timezone.utc)

    # Build masked image
    base = np.full((32, 32, 3), 255, dtype=np.uint8)
    regions = [{"monitor_id": "m1", "x": 8, "y": 8, "width": 10, "height": 10}]
    apply_exclude_region_masks(
        base,
        monitor_id="m1",
        roi_origin_x=0,
        roi_origin_y=0,
        exclude_regions=regions,
    )
    media_dir = tmp_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    masked_path = media_dir / "masked.png"
    Image.fromarray(base).save(masked_path)
    frame_hash = hash_rgb_image(base)

    with db.session() as session:
        session.add(
            CaptureRecord(
                id=excluded_id,
                event_id=excluded_id,
                captured_at=now,
                created_at_utc=now,
                monotonic_ts=1.0,
                image_path=None,
                focus_path=None,
                foreground_process="App",
                foreground_window="Window",
                monitor_id="m1",
                monitor_bounds=[0, 0, 32, 32],
                is_fullscreen=False,
                privacy_flags={"excluded": True},
                frame_hash=None,
                schema_version="v1",
                ocr_status="pending",
            )
        )
        session.add(
            FrameRecord(
                frame_id=excluded_id,
                event_id=None,
                captured_at_utc=now,
                monotonic_ts=1.0,
                monitor_id="m1",
                monitor_bounds=[0, 0, 32, 32],
                app_name="App",
                window_title="Window",
                media_path=None,
                privacy_flags={"excluded": True},
                frame_hash=None,
                excluded=True,
                masked=False,
                schema_version=1,
                created_at=now,
            )
        )
        session.add(
            CaptureRecord(
                id=masked_id,
                event_id=masked_id,
                captured_at=now,
                created_at_utc=now,
                monotonic_ts=1.0,
                image_path=str(masked_path),
                focus_path=None,
                foreground_process="App",
                foreground_window="Window",
                monitor_id="m1",
                monitor_bounds=[0, 0, 32, 32],
                is_fullscreen=False,
                privacy_flags={"masked_regions_applied": True},
                frame_hash=frame_hash,
                schema_version="v1",
                ocr_status="pending",
            )
        )
        session.add(
            FrameRecord(
                frame_id=masked_id,
                event_id=None,
                captured_at_utc=now,
                monotonic_ts=1.0,
                monitor_id="m1",
                monitor_bounds=[0, 0, 32, 32],
                app_name="App",
                window_title="Window",
                media_path=str(masked_path),
                privacy_flags={"masked_regions_applied": True},
                frame_hash=frame_hash,
                excluded=False,
                masked=True,
                schema_version=1,
                created_at=now,
            )
        )

    worker = EventIngestWorker(config, db_manager=db, ocr_processor=_StubExtractor())
    worker.process_batch(limit=5)

    report = {
        "excluded_media_files_found": 0,
        "excluded_artifacts": 0,
        "excluded_spans": 0,
        "excluded_events": 0,
        "masked_leak_pixels": 0,
    }
    if (tmp_dir / "media" / f"{excluded_id}.png").exists():
        report["excluded_media_files_found"] = 1
    with db.session() as session:
        report["excluded_artifacts"] = (
            session.query(ArtifactRecord).filter_by(frame_id=excluded_id).count()
        )
        report["excluded_spans"] = (
            session.query(CitableSpanRecord).filter_by(frame_id=excluded_id).count()
        )
        report["excluded_events"] = (
            session.query(EventRecord).filter_by(event_id=excluded_id).count()
        )

    masked_img = np.asarray(Image.open(masked_path))
    mask_region = masked_img[8:18, 8:18]
    report["masked_leak_pixels"] = int(np.count_nonzero(mask_region))

    output_path = Path("artifacts") / "privacy_report.json"
    _write_report(report, path=output_path)

    if (
        report["excluded_artifacts"]
        or report["excluded_spans"]
        or report["excluded_events"]
        or report["excluded_media_files_found"]
        or report["masked_leak_pixels"] > 0
    ):
        print("Privacy scanner failed; see artifacts/privacy_report.json")
        return 2
    print("Privacy scanner passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
