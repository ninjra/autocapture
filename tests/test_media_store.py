from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from autocapture.config import CaptureConfig, EncryptionConfig
from autocapture.media.store import MediaStore


def test_media_store_roundtrip_roi(tmp_path: Path) -> None:
    capture_config = CaptureConfig(
        staging_dir=tmp_path / "staging",
        data_dir=tmp_path / "data",
    )
    encryption_config = EncryptionConfig(enabled=False)
    store = MediaStore(capture_config, encryption_config)
    image = np.random.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)

    path = store.write_roi(image, dt.datetime.now(dt.timezone.utc), "obs-1")
    assert path is not None
    roundtrip = store.read_image(path)

    assert np.array_equal(image, roundtrip)
