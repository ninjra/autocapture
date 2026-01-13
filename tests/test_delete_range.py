from __future__ import annotations

import datetime as dt
from pathlib import Path

from autocapture.config import DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.deletion import delete_range
from autocapture.storage.models import (
    CaptureRecord,
    EmbeddingRecord,
    EventRecord,
    OCRSpanRecord,
    SegmentRecord,
)


def test_delete_range_removes_rows_and_files(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite"
    db = DatabaseManager(DatabaseConfig(url=f"sqlite:///{db_path}"))
    media_dir = tmp_path / "media"
    media_dir.mkdir()

    now = dt.datetime.now(dt.timezone.utc)
    image_path = media_dir / "capture.webp"
    image_path.write_bytes(b"capture")
    video_path = media_dir / "segment.mp4"
    video_path.write_bytes(b"video")

    with db.session() as session:
        session.add(
            CaptureRecord(
                id="capture-1",
                captured_at=now,
                image_path=str(image_path),
                foreground_process="Docs",
                foreground_window="Notes",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.add(
            EventRecord(
                event_id="capture-1",
                ts_start=now,
                ts_end=None,
                app_name="Docs",
                window_title="Notes",
                url=None,
                domain=None,
                screenshot_path=str(image_path),
                screenshot_hash="hash",
                ocr_text="hello",
                embedding_vector=None,
                tags={},
            )
        )
        session.add(
            OCRSpanRecord(
                capture_id="capture-1",
                span_key="S1",
                start=0,
                end=5,
                text="hello",
                confidence=0.9,
                bbox={},
            )
        )
        session.flush()
        session.add(
            EmbeddingRecord(
                capture_id="capture-1",
                span_key="S1",
                vector=[0.1, 0.2],
                model="test",
                status="done",
            )
        )
        session.add(
            SegmentRecord(
                id="segment-1",
                started_at=now,
                ended_at=now,
                state="done",
                video_path=str(video_path),
                encoder="test",
                frame_count=1,
            )
        )

    counts = delete_range(
        db,
        tmp_path,
        start_utc=now - dt.timedelta(minutes=1),
        end_utc=now + dt.timedelta(minutes=1),
    )
    assert counts.deleted_captures == 1
    assert counts.deleted_events == 1
    assert counts.deleted_segments == 1
    assert counts.deleted_files >= 1

    with db.session() as session:
        assert session.get(CaptureRecord, "capture-1") is None
        assert session.get(EventRecord, "capture-1") is None
        spans = session.query(OCRSpanRecord).all()
        embeddings = session.query(EmbeddingRecord).all()
        segments = session.query(SegmentRecord).all()
        assert spans == []
        assert embeddings == []
        assert segments == []

    assert not image_path.exists()
    assert not video_path.exists()
