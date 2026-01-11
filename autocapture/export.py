"""Export/backup helpers."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy import select

from .config import AppConfig
from .logging_utils import get_logger
from .media.store import MediaStore
from .settings_store import read_settings, update_settings
from .storage.database import DatabaseManager
from .storage.models import EventRecord


def export_capture(
    config: AppConfig,
    *,
    out_path: Path,
    days: int = 90,
    include_media: bool = True,
    decrypt_media: bool = False,
    zip_output: bool = True,
) -> Path:
    log = get_logger("export")
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(days=days)
    data_dir = Path(config.capture.data_dir)
    export_root = out_path
    temp_dir = None
    if zip_output:
        temp_dir = Path(tempfile.mkdtemp(prefix="autocapture-export-"))
        export_root = temp_dir
    export_root.mkdir(parents=True, exist_ok=True)

    db = DatabaseManager(config.database)
    with db.session() as session:
        events = (
            session.execute(
                select(EventRecord)
                .where(EventRecord.ts_start >= cutoff)
                .order_by(EventRecord.ts_start.asc())
            )
            .scalars()
            .all()
        )

    events_path = export_root / "events.jsonl"
    media_manifest: dict[str, Any] = {
        "media_included": include_media,
        "media_encrypted": bool(config.encryption.enabled and not decrypt_media),
        "entries": [],
    }
    media_store = MediaStore(config.capture, config.encryption)

    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            screenshot_path = event.screenshot_path
            rel_path = None
            if screenshot_path:
                rel_path = _relative_path(Path(screenshot_path), data_dir)
            payload = {
                "event_id": event.event_id,
                "ts_start": _to_iso(event.ts_start),
                "ts_end": _to_iso(event.ts_end),
                "app_name": event.app_name,
                "window_title": event.window_title,
                "url": event.url,
                "domain": event.domain,
                "screenshot_path": rel_path or screenshot_path,
                "screenshot_hash": event.screenshot_hash,
                "ocr_text": event.ocr_text,
                "tags": event.tags,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if include_media and screenshot_path:
                source_path = Path(screenshot_path)
                if not source_path.exists():
                    continue
                export_rel = Path(rel_path) if rel_path else Path(source_path.name)
                if decrypt_media and config.encryption.enabled:
                    export_rel = export_rel.with_suffix(".png")
                destination = export_root / export_rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                if decrypt_media and config.encryption.enabled:
                    image = media_store.read_image(source_path)
                    Image.fromarray(image).save(destination, format="PNG")
                else:
                    shutil.copy2(source_path, destination)
                media_manifest["entries"].append(
                    {
                        "source": str(rel_path or source_path),
                        "export_path": str(export_rel),
                        "encrypted": bool(
                            source_path.suffix.endswith(".acenc")
                            and not decrypt_media
                        ),
                    }
                )

    (export_root / "manifest.json").write_text(
        json.dumps(media_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    settings_path = data_dir / "settings.json"
    settings = read_settings(settings_path)
    (export_root / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    redacted_config = _redact_config(config)
    (export_root / "config.json").write_text(
        json.dumps(redacted_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if zip_output:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in export_root.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(export_root))
        shutil.rmtree(export_root, ignore_errors=True)
    log.info("Exported {} events to {}", len(events), out_path)

    update_settings(
        settings_path,
        lambda current: _update_backup_timestamp(current, now),
    )
    return out_path


def _relative_path(path: Path, data_dir: Path) -> str | None:
    try:
        return str(path.relative_to(data_dir))
    except ValueError:
        return None


def _redact_config(config: AppConfig) -> dict[str, Any]:
    data = config.model_dump() if hasattr(config, "model_dump") else config.dict()
    secrets = {
        "api_key",
        "openai_api_key",
        "google_oauth_client_secret",
        "github_token",
        "key_provider",
        "key_name",
    }

    def _redact(value):
        if isinstance(value, dict):
            return {
                key: ("***" if key in secrets and value.get(key) else _redact(val))
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [_redact(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    return _redact(data)


def _update_backup_timestamp(settings: dict[str, Any], now: dt.datetime) -> dict[str, Any]:
    backup = settings.get("backup")
    if not isinstance(backup, dict):
        backup = {}
    backup["last_export_at_utc"] = _to_iso(now)
    settings["backup"] = backup
    return settings


def _to_iso(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.isoformat()
