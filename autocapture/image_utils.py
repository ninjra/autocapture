"""Image helper utilities for hashing and normalization."""

from __future__ import annotations

import hashlib

import numpy as np


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape HxWx3")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


def hash_rgb_image(image: np.ndarray) -> str:
    rgb = ensure_rgb(image)
    height, width, _ = rgb.shape
    hasher = hashlib.sha256()
    hasher.update(f"{width}x{height}".encode("utf-8"))
    hasher.update(rgb.tobytes())
    return hasher.hexdigest()
