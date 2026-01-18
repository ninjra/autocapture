from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from autocapture.config import OCRConfig
from autocapture.vision.paddle_layout import PaddleLayoutExtractor


def test_ppstructure_missing_dependency(tmp_path) -> None:
    if importlib.util.find_spec("paddleocr") is not None:
        pytest.skip("paddleocr installed; skip fallback test")
    config = OCRConfig()
    config.paddle_ppstructure_enabled = True
    config.paddle_ppstructure_model_dir = tmp_path
    extractor = PaddleLayoutExtractor(config)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    result = extractor.extract(image)
    assert result is not None
    assert result.tags.get("status") == "missing_dependency"
