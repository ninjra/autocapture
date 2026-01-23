"""Storage stats service for UX surfaces."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

from sqlalchemy.engine.url import make_url

from ..config import AppConfig
from ..storage.retention import RetentionManager
from .models import StorageStatsResponse


class StorageService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._cache: dict[str, object] | None = None
        self._cache_ts = 0.0
        self._cache_ttl_s = 30.0

    def stats(self) -> StorageStatsResponse:
        now = time.monotonic()
        if self._cache and now - self._cache_ts < self._cache_ttl_s:
            cached = self._cache["payload"]
            age = now - self._cache_ts
            return cached.model_copy(update={"cache_hit": True, "cache_age_s": age})

        data_dir = Path(self._config.capture.data_dir)
        media_dir = data_dir / "media"
        staging_dir = Path(self._config.capture.staging_dir)
        db_path = _resolve_db_path(self._config.database.url)
        media_bytes = _folder_size(media_dir)
        staging_bytes = _folder_size(staging_dir)
        db_bytes = db_path.stat().st_size if db_path and db_path.exists() else 0
        total_bytes = int(media_bytes + staging_bytes + db_bytes)
        free_bytes = None
        try:
            free_bytes = int(_disk_free_bytes(data_dir))
        except Exception:
            free_bytes = None

        payload = StorageStatsResponse(
            data_dir=str(data_dir),
            media_dir=str(media_dir),
            staging_dir=str(staging_dir),
            db_path=str(db_path) if db_path else None,
            media_bytes=int(media_bytes),
            staging_bytes=int(staging_bytes),
            db_bytes=int(db_bytes),
            total_bytes=int(total_bytes),
            free_bytes=free_bytes,
            collected_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            cache_hit=False,
            cache_age_s=0.0,
        )
        self._cache = {"payload": payload}
        self._cache_ts = now
        return payload


def _resolve_db_path(url: str) -> Path | None:
    try:
        parsed = make_url(url)
    except Exception:
        return None
    if not parsed.drivername.startswith("sqlite"):
        return None
    if not parsed.database:
        return None
    path = Path(parsed.database)
    return path


def _folder_size(path: Path) -> int:
    if not path.exists():
        return 0
    return int(RetentionManager._folder_size_bytes(path))


def _disk_free_bytes(path: Path) -> int:
    try:
        import shutil

        return int(shutil.disk_usage(path).free)
    except Exception:
        return 0
