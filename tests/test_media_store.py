from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autocapture.config import CaptureConfig, EncryptionConfig
from autocapture.media.store import MediaStore


def test_media_store_roundtrip_roi(tmp_path: Path) -> None:
    capture_config = CaptureConfig(
        staging_dir=tmp_path / "staging",
        data_dir=tmp_path / "data",
        staging_min_free_mb=0,
        data_min_free_mb=0,
    )
    encryption_config = EncryptionConfig(enabled=False)
    store = MediaStore(capture_config, encryption_config)
    image = np.random.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)

    path = store.write_roi(image, dt.datetime.now(dt.timezone.utc), "obs-1")
    assert path is not None
    roundtrip = store.read_image(path)

    assert np.array_equal(image, roundtrip)


def test_media_store_encryption_toggle_read(tmp_path: Path) -> None:
    key_path = tmp_path / "key.bin"
    capture_config = CaptureConfig(
        staging_dir=tmp_path / "staging",
        data_dir=tmp_path / "data",
        staging_min_free_mb=0,
        data_min_free_mb=0,
    )
    encryption_config = EncryptionConfig(
        enabled=True,
        key_provider=f"file:{key_path}",
    )
    store = MediaStore(capture_config, encryption_config)
    image = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
    path = store.write_roi(image, dt.datetime.now(dt.timezone.utc), "obs-enc")
    assert path is not None
    assert path.suffix.endswith(".acenc")

    read_store = MediaStore(
        capture_config,
        EncryptionConfig(enabled=False, key_provider=f"file:{key_path}"),
    )
    roundtrip = read_store.read_image(path)
    assert np.array_equal(image, roundtrip)


def test_media_store_cleanup_on_save_failure(tmp_path: Path, monkeypatch) -> None:
    capture_config = CaptureConfig(
        staging_dir=tmp_path / "staging",
        data_dir=tmp_path / "data",
        staging_min_free_mb=0,
        data_min_free_mb=0,
    )
    encryption_config = EncryptionConfig(enabled=False)
    store = MediaStore(capture_config, encryption_config)
    image = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)

    def _fail_save(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", _fail_save, raising=True)

    with pytest.raises(OSError):
        store.write_roi(image, dt.datetime.now(dt.timezone.utc), "obs-fail")

    staging_path = capture_config.staging_dir / "roi_obs-fail.tmp"
    assert not staging_path.exists()


def test_media_store_disk_guard_blocks_write(tmp_path: Path, monkeypatch) -> None:
    capture_config = CaptureConfig(
        staging_dir=tmp_path / "staging",
        data_dir=tmp_path / "data",
        staging_min_free_mb=10,
        data_min_free_mb=10,
    )
    encryption_config = EncryptionConfig(enabled=False)
    store = MediaStore(capture_config, encryption_config)

    class DiskUsage:
        def __init__(self, total: int, used: int, free: int) -> None:
            self.total = total
            self.used = used
            self.free = free

    def _fake_disk_usage(_path: Path) -> DiskUsage:
        return DiskUsage(total=50 * 1024 * 1024, used=49 * 1024 * 1024, free=1024)

    monkeypatch.setattr("autocapture.media.store.shutil.disk_usage", _fake_disk_usage)

    image = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
    path = store.write_roi(image, dt.datetime.now(dt.timezone.utc), "obs-low-disk")

    assert path is None
    staging_path = capture_config.staging_dir / "roi_obs-low-disk.tmp"
    assert not staging_path.exists()
