"""Perceptual hash based duplicate detection."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class DuplicateResult:
    is_duplicate: bool
    distance: float


@dataclass(frozen=True)
class _Signature:
    ts: float
    phash: np.ndarray
    gray: np.ndarray


class DuplicateDetector:
    """Thread-safe perceptual hash cache to discard redundant frames."""

    def __init__(
        self,
        *,
        threshold: float = 0.02,
        window_s: float = 10.0,
        max_items: int = 16,
        pixel_threshold: float = 2.5,
    ) -> None:
        self._threshold = threshold
        self._window_s = window_s
        self._max_items = max_items
        self._pixel_threshold = pixel_threshold
        self._lock = threading.Lock()
        self._recent: deque[_Signature] = deque(maxlen=max_items)

    def update(self, image: Image.Image) -> DuplicateResult:
        """Return whether the provided frame is a duplicate."""

        current = self._signature(image)
        now = time.monotonic()
        with self._lock:
            while self._recent and now - self._recent[0].ts > self._window_s:
                self._recent.popleft()
            for prior in self._recent:
                phash_dist = float(np.mean(current.phash != prior.phash))
                pixel_mae = float(np.mean(np.abs(current.gray - prior.gray)))
                if phash_dist <= self._threshold and pixel_mae <= self._pixel_threshold:
                    return DuplicateResult(is_duplicate=True, distance=phash_dist)
            self._recent.append(current)
            return DuplicateResult(is_duplicate=False, distance=1.0)

    @staticmethod
    def _signature(image: Image.Image) -> _Signature:
        phash = DuplicateDetector._phash(image)
        gray = DuplicateDetector._downsample_gray(image)
        return _Signature(ts=time.monotonic(), phash=phash, gray=gray)

    @staticmethod
    def _downsample_gray(image: Image.Image, size: int = 64) -> np.ndarray:
        gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(gray, dtype=np.float32)

    @staticmethod
    def _phash(image: Image.Image, hash_size: int = 16) -> np.ndarray:
        image = image.convert("L").resize((hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32)
        dct = np.fft.fft2(pixels)
        dctlow = dct[:hash_size, :hash_size]
        med = np.median(np.abs(dctlow))
        return dctlow > med
