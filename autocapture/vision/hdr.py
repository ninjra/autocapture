"""HDR detection/tone mapping helpers (best-effort)."""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_hdr_tone_mapping(
    image: np.ndarray, *, enabled: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    tags: dict[str, Any] = {
        "enabled": bool(enabled),
        "detected": False,
        "mode": "disabled" if not enabled else "noop",
        "color_space": "sRGB",
    }
    if not enabled:
        return image, tags
    if image.dtype != np.uint8 or float(np.max(image)) > 255.0:
        tags["detected"] = True
        tags["mode"] = "scaled_to_srgb"
        max_val = float(np.max(image))
        if max_val <= 0.0:
            scaled = np.zeros_like(image, dtype=np.uint8)
        else:
            scaled = (image.astype(np.float32) / max_val * 255.0).clip(0, 255).astype(np.uint8)
        return scaled, tags
    tags["mode"] = "noop"
    return image, tags
