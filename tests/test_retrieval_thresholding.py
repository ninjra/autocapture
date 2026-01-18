import datetime as dt
from pathlib import Path

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.memory.retrieval import RetrievalService
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord


class _EmptyLexical:
    def search(self, *_args, **_kwargs):
        return []


class _EmptyVector:
    def search(self, *_args, **_kwargs):
        return []


def test_thresholding_can_return_no_evidence(tmp_path: Path) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.features.enable_thresholding = True
    config.retrieval.lexical_min_score = 0.5
    config.retrieval.dense_min_score = 0.5
    db = DatabaseManager(config.database)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="evt-1",
                ts_start=dt.datetime.now(dt.timezone.utc),
                ts_end=None,
                app_name="App",
                window_title="Title",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash=None,
                ocr_text="alpha beta",
                tags={},
            )
        )
    retrieval = RetrievalService(db, config)
    retrieval._lexical = _EmptyLexical()  # type: ignore[attr-defined]
    retrieval._vector = _EmptyVector()  # type: ignore[attr-defined]
    batch = retrieval.retrieve("alpha", None, None, limit=3)
    assert batch.no_evidence is True
