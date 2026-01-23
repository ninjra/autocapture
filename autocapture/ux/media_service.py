"""Media resolution + thumbnail handling for UX surfaces."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..config import AppConfig
from ..encryption import EncryptionManager
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import EventRecord


class MediaNotFoundError(ValueError):
    pass


class MediaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class MediaResult:
    body: bytes | None
    path: Path | None
    media_type: str
    headers: dict[str, str]


class MediaService:
    def __init__(
        self,
        config: AppConfig,
        db: DatabaseManager,
        encryption_mgr: EncryptionManager,
    ) -> None:
        self._config = config
        self._db = db
        self._encryption_mgr = encryption_mgr
        self._log = get_logger("ux.media")
        self._cache_dir = Path(config.capture.data_dir) / "cache" / "thumbs"
        self._thumb_width = int(config.capture.thumbnail_width)
        self._cache_max_entries = 5000
        self._cache_max_bytes = 2 * 1024 * 1024 * 1024
        self._cache_ttl_s = 7 * 24 * 3600

    def fetch(self, *, event_id: str, kind: str, variant: str) -> MediaResult:
        kind = (kind or "").strip().lower()
        variant = (variant or "").strip().lower()
        if kind not in {"screenshot", "focus"}:
            raise MediaValidationError("Unsupported media kind")
        if variant not in {"full", "thumb"}:
            raise MediaValidationError("variant must be full or thumb")

        event = self._load_event(event_id)
        path_value = self._resolve_path_value(event, kind)
        if not path_value:
            raise MediaNotFoundError("Media not found")
        resolved = self._resolve_path(path_value)
        if variant == "full":
            return self._serve_full(resolved)
        return self._serve_thumb(event_id=event_id, kind=kind, source=resolved)

    def _load_event(self, event_id: str) -> EventRecord:
        with self._db.session() as session:
            event = session.get(EventRecord, event_id)
        if not event:
            raise MediaNotFoundError("Event not found")
        return event

    def _resolve_path_value(self, event: EventRecord, kind: str) -> str | None:
        if kind == "screenshot":
            return event.screenshot_path
        if event.focus_path:
            return event.focus_path
        tags = event.tags or {}
        if isinstance(tags, dict):
            capture = tags.get("capture")
            if isinstance(capture, dict):
                focus_path = capture.get("focus_path")
                if isinstance(focus_path, str) and focus_path.strip():
                    return focus_path
        return None

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = Path(self._config.capture.data_dir) / path
        try:
            root = Path(self._config.capture.data_dir).resolve()
            resolved = path.resolve()
            if root not in resolved.parents and resolved != root:
                raise MediaValidationError("Invalid media path")
            path = resolved
        except FileNotFoundError:
            pass
        if not path.exists():
            raise MediaNotFoundError("Media file missing")
        return path

    def _serve_full(self, path: Path) -> MediaResult:
        headers = {"Cache-Control": "private, max-age=60"}
        if path.suffix == ".acenc":
            with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                self._encryption_mgr.decrypt_file(path, tmp_path)
                data = tmp_path.read_bytes()
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            return MediaResult(body=data, path=None, media_type="image/webp", headers=headers)
        return MediaResult(body=None, path=path, media_type="image/webp", headers=headers)

    def _serve_thumb(self, *, event_id: str, kind: str, source: Path) -> MediaResult:
        headers = {"Cache-Control": "private, max-age=120"}
        cache_path = self._cache_path(event_id=event_id, kind=kind, source=source)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return MediaResult(body=cached, path=None, media_type="image/webp", headers=headers)
        thumb_bytes = self._render_thumb(source)
        if self._cache_enabled():
            self._write_cache(cache_path, thumb_bytes)
        return MediaResult(body=thumb_bytes, path=None, media_type="image/webp", headers=headers)

    def _render_thumb(self, source: Path) -> bytes:
        image, cleanup = self._open_image(source)
        try:
            image = image.convert("RGB")
            width = max(self._thumb_width, 1)
            if image.width > width:
                height = max(int(image.height * width / image.width), 1)
                image = image.resize((width, height), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP", quality=75, method=6)
            return buffer.getvalue()
        finally:
            try:
                cleanup()
            except Exception:
                pass

    def _open_image(self, source: Path):
        if source.suffix != ".acenc":
            image = Image.open(source)
            image.load()
            return image, lambda: None
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        def _cleanup():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            self._encryption_mgr.decrypt_file(source, tmp_path)
            image = Image.open(tmp_path)
            image.load()
            return image, _cleanup
        except Exception:
            _cleanup()
            raise

    def _cache_enabled(self) -> bool:
        return self._cache_max_entries > 0 and self._cache_max_bytes > 0

    def _cache_path(self, *, event_id: str, kind: str, source: Path) -> Path:
        mtime_ns = int(source.stat().st_mtime_ns)
        key = f"{event_id}:{kind}:{self._thumb_width}:{mtime_ns}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        suffix = ".acenc" if self._encryption_mgr.enabled else ".webp"
        return self._cache_dir / f"{digest}{suffix}"

    def _read_cache(self, path: Path) -> bytes | None:
        if not self._cache_enabled():
            return None
        if not path.exists():
            return None
        if self._cache_ttl_s > 0:
            age = time.time() - path.stat().st_mtime
            if age > self._cache_ttl_s:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
                return None
        try:
            os.utime(path, None)
        except Exception:
            pass
        if path.suffix == ".acenc":
            with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                self._encryption_mgr.decrypt_file(path, tmp_path)
                return tmp_path.read_bytes()
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
        return path.read_bytes()

    def _write_cache(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            if self._encryption_mgr.enabled:
                with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp:
                    raw_path = Path(tmp.name)
                try:
                    raw_path.write_bytes(data)
                    self._encryption_mgr.encrypt_file(raw_path, tmp_path)
                finally:
                    try:
                        raw_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                tmp_path.write_bytes(data)
            tmp_path.replace(path)
        except Exception as exc:
            self._log.warning("Thumb cache write failed: {}", exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        self._prune_cache()

    def _prune_cache(self) -> None:
        if not self._cache_enabled():
            return
        if not self._cache_dir.exists():
            return
        entries: list[tuple[float, Path, int]] = []
        total = 0
        now = time.time()
        for item in self._cache_dir.iterdir():
            if not item.is_file():
                continue
            try:
                stat = item.stat()
            except FileNotFoundError:
                continue
            if self._cache_ttl_s > 0 and now - stat.st_mtime > self._cache_ttl_s:
                try:
                    item.unlink(missing_ok=True)
                except Exception:
                    pass
                continue
            entries.append((stat.st_mtime, item, int(stat.st_size)))
            total += int(stat.st_size)
        if not entries:
            return
        entries.sort(key=lambda entry: entry[0])
        removed = 0
        while entries and (
            len(entries) > self._cache_max_entries or total > self._cache_max_bytes
        ):
            _, path, size = entries.pop(0)
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            total -= size
            removed += 1
        if removed:
            self._log.debug("Pruned {} cached thumbs", removed)
