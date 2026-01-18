import datetime as dt

from autocapture.config import DatabaseConfig
from autocapture.indexing.lexical_index import LexicalIndex
from autocapture.storage.database import DatabaseManager
from autocapture.storage.integrity import scan_integrity
from autocapture.storage.models import EventRecord


class _StubVectorIndex:
    def list_event_ids(self):
        return ["orphan-vector"]


def test_integrity_scan_detects_orphans(tmp_path):
    db_path = tmp_path / "scan.db"
    db = DatabaseManager(DatabaseConfig(url=f"sqlite:///{db_path}"))
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="event-1",
                ts_start=now,
                ts_end=None,
                app_name="App",
                window_title="Title",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash=None,
                ocr_text="hello world",
                tags={},
            )
        )
    LexicalIndex(db).upsert_event(
        EventRecord(
            event_id="orphan-event",
            ts_start=now,
            ts_end=None,
            app_name="App",
            window_title="Title",
            url=None,
            domain=None,
            screenshot_path=None,
            screenshot_hash=None,
            ocr_text="orphan text",
            tags={},
        )
    )
    report = scan_integrity(db, vector_index=_StubVectorIndex())
    assert report.orphan_fts >= 1
    assert report.orphan_vectors == 1


def test_integrity_scan_clean_fixture(tmp_path):
    db_path = tmp_path / "clean.db"
    db = DatabaseManager(DatabaseConfig(url=f"sqlite:///{db_path}"))
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="event-clean",
                ts_start=now,
                ts_end=None,
                app_name="App",
                window_title="Title",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash=None,
                ocr_text="clean text",
                tags={},
            )
        )
    LexicalIndex(db).upsert_event(
        EventRecord(
            event_id="event-clean",
            ts_start=now,
            ts_end=None,
            app_name="App",
            window_title="Title",
            url=None,
            domain=None,
            screenshot_path=None,
            screenshot_hash=None,
            ocr_text="clean text",
            tags={},
        )
    )
    report = scan_integrity(db)
    assert report.orphan_fts == 0
    assert report.orphan_spans == 0
