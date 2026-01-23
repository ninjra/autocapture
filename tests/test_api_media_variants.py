from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from PIL import Image

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord


def _make_app(tmp_path: Path) -> tuple[object, DatabaseManager]:
    config = AppConfig()
    config.database = DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}")
    config.capture.data_dir = tmp_path
    config.encryption.enabled = False
    config.embed.text_model = "local-test"
    db = DatabaseManager(config.database)
    app = create_app(config, db_manager=db)
    return app, db


def _write_webp(path: Path, *, size: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), color=(20, 80, 140))
    image.save(path, format="WEBP")


@pytest.mark.anyio
async def test_api_screenshot_variants(tmp_path: Path, async_client_factory) -> None:
    app, db = _make_app(tmp_path)
    shot_path = tmp_path / "media" / "event-1.webp"
    _write_webp(shot_path)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="event-1",
                ts_start=now,
                ts_end=None,
                app_name="Notes",
                window_title="Example",
                url=None,
                domain=None,
                screenshot_path=str(Path("media") / "event-1.webp"),
                screenshot_hash="hash-1",
                ocr_text="hello",
                embedding_vector=None,
                tags={},
            )
        )

    async with async_client_factory(app) as client:
        full = await client.get("/api/screenshot/event-1?variant=full")
        assert full.status_code == 200
        assert full.headers["content-type"].startswith("image/webp")

        thumb = await client.get("/api/screenshot/event-1?variant=thumb")
        assert thumb.status_code == 200
        assert thumb.headers["content-type"].startswith("image/webp")


@pytest.mark.anyio
async def test_api_focus_variants(tmp_path: Path, async_client_factory) -> None:
    app, db = _make_app(tmp_path)
    shot_path = tmp_path / "media" / "event-2.webp"
    focus_path = tmp_path / "media" / "event-2-focus.webp"
    _write_webp(shot_path, size=160)
    _write_webp(focus_path, size=80)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="event-2",
                ts_start=now,
                ts_end=None,
                app_name="Browser",
                window_title="Example",
                url=None,
                domain=None,
                screenshot_path=str(Path("media") / "event-2.webp"),
                focus_path=str(Path("media") / "event-2-focus.webp"),
                screenshot_hash="hash-2",
                ocr_text="hello",
                embedding_vector=None,
                tags={},
            )
        )

    async with async_client_factory(app) as client:
        full = await client.get("/api/focus/event-2?variant=full")
        assert full.status_code == 200
        assert full.headers["content-type"].startswith("image/webp")

        thumb = await client.get("/api/focus/event-2?variant=thumb")
        assert thumb.status_code == 200
        assert thumb.headers["content-type"].startswith("image/webp")

        legacy = await client.get("/api/screenshot/event-2?variant=focus")
        assert legacy.status_code == 200
        assert legacy.headers["content-type"].startswith("image/webp")
