"""Media storage with staging, atomic writes, and optional encryption."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..config import CaptureConfig, EncryptionConfig
from ..encryption import EncryptionManager
from ..image_utils import ensure_rgb
from ..fs_utils import atomic_publish, safe_unlink
from ..logging_utils import get_logger


class MediaStore:
    def __init__(self, capture_config: CaptureConfig, encryption_config: EncryptionConfig) -> None:
        self._capture_config = capture_config
        self._encryption_config = encryption_config
        self._encryption = EncryptionManager(encryption_config)
        self._log = get_logger("media.store")
        self._staging_dir = Path(capture_config.staging_dir)
        self._data_dir = Path(capture_config.data_dir)
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def write_roi(self, image: np.ndarray, timestamp, observation_id: str) -> Optional[Path]:
        if not self._has_required_space():
            self._log.warning("Disk quota exceeded; dropping ROI {}", observation_id)
            return None
        date_prefix = timestamp.strftime("%Y/%m/%d")
        final_dir = self._data_dir / "media" / "roi" / date_prefix
        final_name = f"{timestamp.strftime('%H%M%S_%f')}_{observation_id}.webp"
        final_path = final_dir / final_name
        if self._encryption.enabled:
            final_path = final_path.with_suffix(final_path.suffix + ".acenc")

        staging_path = self._staging_dir / f"roi_{observation_id}.tmp"
        image = ensure_rgb(image)
        pil = Image.fromarray(np.ascontiguousarray(image))
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pil.save(staging_path, format="WEBP", lossless=True, quality=100, method=6)
            final_dir.mkdir(parents=True, exist_ok=True)
            if self._encryption.enabled:
                atomic_publish(
                    staging_path,
                    final_path,
                    writer=self._encryption.encrypt_file,
                )
            else:
                atomic_publish(
                    staging_path,
                    final_path,
                    writer=_copy_file,
                )
        except Exception:
            safe_unlink(staging_path)
            raise
        return final_path

    def reserve_video_paths(self, timestamp, segment_id: str) -> tuple[Path, Path]:
        date_prefix = timestamp.strftime("%Y/%m/%d")
        final_dir = self._data_dir / "media" / "video" / date_prefix
        final_name = f"segment_{segment_id}.mp4"
        final_path = final_dir / final_name
        if self._encryption.enabled:
            final_path = final_path.with_suffix(final_path.suffix + ".acenc")
        staging_dir = self._staging_dir / "video" / date_prefix
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_path = staging_dir / final_name
        return staging_path, final_path

    def finalize_video(self, staging_path: Path, final_path: Path) -> Path:
        if not staging_path.exists():
            return final_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if self._encryption.enabled:
            atomic_publish(
                staging_path,
                final_path,
                writer=self._encryption.encrypt_file,
            )
        else:
            atomic_publish(
                staging_path,
                final_path,
                writer=_copy_file,
            )
        return final_path

    def read_image(self, path: Path) -> np.ndarray:
        if not path.suffix.endswith(".acenc"):
            with path.open("rb") as handle:
                image = Image.open(handle)
                image = image.convert("RGB")
                return np.array(image)
        if self._encryption.enabled:
            encryption = self._encryption
        else:
            override = (
                self._encryption_config.model_copy(update={"enabled": True})
                if hasattr(self._encryption_config, "model_copy")
                else self._encryption_config.copy(update={"enabled": True})
            )
            encryption = EncryptionManager(override)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            encryption.decrypt_file(path, tmp_path)
            with tmp_path.open("rb") as handle:
                image = Image.open(handle)
                image = image.convert("RGB")
                return np.array(image)
        except Exception as exc:
            raise RuntimeError("Failed to decrypt image; check encryption settings.") from exc
        finally:
            safe_unlink(tmp_path)

    def _has_required_space(self) -> bool:
        try:
            staging_usage = shutil.disk_usage(self._staging_dir)
            data_usage = shutil.disk_usage(self._data_dir)
        except FileNotFoundError:
            return False
        min_staging = self._capture_config.staging_min_free_mb * 1024 * 1024
        min_data = self._capture_config.data_min_free_mb * 1024 * 1024
        if min_staging >= staging_usage.total:
            min_staging = 0
        if min_data >= data_usage.total:
            min_data = 0
        return staging_usage.free >= min_staging and data_usage.free >= min_data


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
