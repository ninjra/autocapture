import datetime as dt
from pathlib import Path

from autocapture.config import DatabaseConfig
from autocapture.indexing.lexical_index import LexicalIndex
from autocapture.indexing.pruner import IndexPruner
from autocapture.storage.database import DatabaseManager
from autocapture.storage.deletion import delete_range
from autocapture.storage.models import CaptureRecord, EventRecord


def test_delete_range_prunes_fts(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite"
    db = DatabaseManager(DatabaseConfig(url=f"sqlite:///{db_path}"))
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            CaptureRecord(
                id="cap-1",
                captured_at=now,
                image_path=None,
                foreground_process="Docs",
                foreground_window="Notes",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.add(
            EventRecord(
                event_id="cap-1",
                ts_start=now,
                ts_end=None,
                app_name="Docs",
                window_title="Notes",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash=None,
                ocr_text="hello keyword",
                tags={},
            )
        )
    lexical = LexicalIndex(db)
    with db.session() as session:
        event = session.get(EventRecord, "cap-1")
    assert event is not None
    lexical.upsert_event(event)
    pruner = IndexPruner(db)
    delete_range(
        db,
        tmp_path,
        start_utc=now - dt.timedelta(minutes=1),
        end_utc=now + dt.timedelta(minutes=1),
        index_pruner=pruner,
    )
    hits = lexical.search("keyword", limit=5)
    assert hits == []
