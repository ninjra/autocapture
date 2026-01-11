from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.export import export_capture
from autocapture.settings_store import read_settings
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord


def test_export_writes_zip_bundle(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "autocapture.db"
    config = AppConfig(
        capture={"data_dir": data_dir, "staging_dir": data_dir / "staging"},
        database=DatabaseConfig(url=f"sqlite:///{db_path}"),
        embed={"text_model": "local-test"},
        encryption={"enabled": False},
    )
    db = DatabaseManager(config.database)

    media_dir = data_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = media_dir / "shot.png"
    screenshot_path.write_bytes(b"fake-image")

    with db.session() as session:
        session.add(
            EventRecord(
                event_id="event-1",
                ts_start=dt.datetime.now(dt.timezone.utc),
                ts_end=None,
                app_name="test",
                window_title="window",
                url=None,
                domain=None,
                screenshot_path=str(screenshot_path),
                screenshot_hash="hash",
                ocr_text="hello",
                embedding_vector=None,
                embedding_status="pending",
                embedding_model=None,
                tags={},
            )
        )

    out_path = tmp_path / "export.zip"
    export_capture(
        config,
        out_path=out_path,
        days=1,
        include_media=True,
        decrypt_media=False,
        zip_output=True,
    )
    assert out_path.exists()
    with zipfile.ZipFile(out_path, "r") as zf:
        names = set(zf.namelist())
    assert "events.jsonl" in names
    assert "config.json" in names
    assert "settings.json" in names
    assert "manifest.json" in names
    assert "media/shot.png" in names

    settings = read_settings(data_dir / "settings.json")
    assert settings.get("backup", {}).get("last_export_at_utc")
