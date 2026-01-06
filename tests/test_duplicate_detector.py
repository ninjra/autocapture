from __future__ import annotations

import numpy as np
from PIL import Image

from autocapture.capture.duplicate import DuplicateDetector


def _image_from_value(value: int) -> Image.Image:
    arr = np.full((32, 32, 3), value, dtype=np.uint8)
    return Image.fromarray(arr)


def test_duplicate_detector_handles_aba_within_window(monkeypatch) -> None:
    detector = DuplicateDetector(window_s=10.0)
    timeline = iter([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    monkeypatch.setattr(
        "autocapture.capture.duplicate.time.monotonic", lambda: next(timeline)
    )

    img_a = _image_from_value(10)
    img_b = _image_from_value(200)

    assert not detector.update(img_a).is_duplicate
    assert not detector.update(img_b).is_duplicate
    assert detector.update(img_a).is_duplicate


def test_duplicate_detector_respects_window(monkeypatch) -> None:
    detector = DuplicateDetector(window_s=1.0)
    timeline = iter([0.0, 0.0, 0.5, 0.5, 2.0, 2.0])
    monkeypatch.setattr(
        "autocapture.capture.duplicate.time.monotonic", lambda: next(timeline)
    )

    img_a = _image_from_value(10)
    img_b = _image_from_value(200)

    assert not detector.update(img_a).is_duplicate
    assert not detector.update(img_b).is_duplicate
    assert not detector.update(img_a).is_duplicate


def test_duplicate_detector_avoids_false_duplicates(monkeypatch) -> None:
    detector = DuplicateDetector(threshold=1.0, pixel_threshold=0.1)
    timeline = iter([0.0, 0.0, 1.0, 1.0])
    monkeypatch.setattr(
        "autocapture.capture.duplicate.time.monotonic", lambda: next(timeline)
    )

    img_a = _image_from_value(10)
    img_b = _image_from_value(40)

    assert not detector.update(img_a).is_duplicate
    assert not detector.update(img_b).is_duplicate
