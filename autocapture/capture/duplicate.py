"""Perceptual hash based duplicate detection."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class DuplicateResult:
    is_duplicate: bool
    distance: float


class DuplicateDetector:
    """Thread-safe perceptual hash cache to discard redundant frames."""

    def __init__(self, threshold: float = 0.02) -> None:
        self._threshold = threshold
        self._lock = threading.Lock()
        self._previous_hash: Optional[np.ndarray] = None

    def update(self, image: Image.Image) -> DuplicateResult:
        """Return whether the provided frame is a duplicate."""

        current = self._phash(image)
        with self._lock:
            if self._previous_hash is None:
                self._previous_hash = current
                return DuplicateResult(is_duplicate=False, distance=1.0)

            distance = float(np.mean(current != self._previous_hash))
            if distance <= self._threshold:
                return DuplicateResult(is_duplicate=True, distance=distance)

            self._previous_hash = current
            return DuplicateResult(is_duplicate=False, distance=distance)

    @staticmethod
    def _phash(image: Image.Image, hash_size: int = 16) -> np.ndarray:
        image = image.convert("L").resize((hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32)
        dct = np.fft.fft2(pixels)
        dctlow = dct[:hash_size, :hash_size]
        med = np.median(np.abs(dctlow))
        return dctlow > med
