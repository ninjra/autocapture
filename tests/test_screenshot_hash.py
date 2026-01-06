from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from autocapture.config import CaptureConfig, EncryptionConfig
from autocapture.encryption import EncryptionManager
from autocapture.image_utils import hash_rgb_image
from autocapture.media.store import MediaStore


def _write_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def test_screenshot_hash_deterministic_with_encryption(tmp_path: Path) -> None:
    capture_config = CaptureConfig(
        staging_dir=tmp_path / "staging",
        data_dir=tmp_path / "data",
    )
    encryption_config = EncryptionConfig(
        enabled=True, key_provider=f"file:{tmp_path / 'key.bin'}"
    )
    manager = EncryptionManager(encryption_config)
    store = MediaStore(capture_config, encryption_config)

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    source = tmp_path / "source.png"
    _write_image(source, image)

    enc_a = tmp_path / "enc_a.acenc"
    enc_b = tmp_path / "enc_b.acenc"
    manager.encrypt_file(source, enc_a)
    manager.encrypt_file(source, enc_b)

    hash_a = hash_rgb_image(store.read_image(enc_a))
    hash_b = hash_rgb_image(store.read_image(enc_b))

    assert hash_a == hash_b

    image[0, 0] = [255, 0, 0]
    _write_image(source, image)
    enc_c = tmp_path / "enc_c.acenc"
    manager.encrypt_file(source, enc_c)
    hash_c = hash_rgb_image(store.read_image(enc_c))
    assert hash_c != hash_a
