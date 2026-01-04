"""Media storage with staging, atomic writes, and optional encryption."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..config import CaptureConfig, EncryptionConfig
from ..encryption import EncryptionManager
from ..logging_utils import get_logger


class MediaStore:
    def __init__(self, capture_config: CaptureConfig, encryption_config: EncryptionConfig) -> None:
        self._capture_config = capture_config
        self._encryption = EncryptionManager(encryption_config)
        self._log = get_logger("media.store")
        self._staging_dir = Path(capture_config.staging_dir)
        self._data_dir = Path(capture_config.data_dir)
        self._staging_dir.mkdir(parents=True, exist_ok=True)

    def write_roi(self, image: np.ndarray, timestamp, observation_id: str) -> Optional[Path]:
        if not self._has_staging_space():
            self._log.warning("Staging quota exceeded; dropping ROI %s", observation_id)
            return None
        date_prefix = timestamp.strftime("%Y/%m/%d")
        final_dir = self._data_dir / "media" / "roi" / date_prefix
        final_name = f"{timestamp.strftime('%H%M%S_%f')}_{observation_id}.webp"
        final_path = final_dir / final_name
        if self._encryption.enabled:
            final_path = final_path.with_suffix(final_path.suffix + ".acenc")

        staging_path = self._staging_dir / f"roi_{observation_id}.tmp"
        rgb = image[:, :, ::-1]
        pil = Image.fromarray(np.ascontiguousarray(rgb))
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(staging_path, format="WEBP", quality=80, method=6)
        self._fsync_file(staging_path)

        final_dir.mkdir(parents=True, exist_ok=True)
        if self._encryption.enabled:
            encrypted_tmp = self._staging_dir / f"roi_{observation_id}.enc"
            self._encryption.encrypt_file(staging_path, encrypted_tmp)
            self._fsync_file(encrypted_tmp)
            encrypted_tmp.replace(final_path)
            staging_path.unlink(missing_ok=True)
        else:
            staging_path.replace(final_path)
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
            encrypted_tmp = self._staging_dir / f"segment_{staging_path.stem}.enc"
            self._encryption.encrypt_file(staging_path, encrypted_tmp)
            self._fsync_file(encrypted_tmp)
            encrypted_tmp.replace(final_path)
            staging_path.unlink(missing_ok=True)
        else:
            staging_path.replace(final_path)
        return final_path

    def read_image(self, path: Path) -> np.ndarray:
        if not self._encryption.enabled or not path.suffix.endswith(".acenc"):
            with path.open("rb") as handle:
                image = Image.open(handle)
                image = image.convert("RGB")
                return np.array(image)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self._encryption.decrypt_file(path, tmp_path)
            with tmp_path.open("rb") as handle:
                image = Image.open(handle)
                image = image.convert("RGB")
                return np.array(image)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _has_staging_space(self) -> bool:
        try:
            usage = shutil.disk_usage(self._staging_dir)
        except FileNotFoundError:
            return False
        return usage.free > 100 * 1024 * 1024

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
