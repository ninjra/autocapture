"""Image tiling utilities for VLM extraction."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisionTile:
    index: int
    bbox_norm: tuple[float, float, float, float]
    image_bytes: bytes
    kind: str


def build_tiles(
    image: np.ndarray,
    *,
    tiles_x: int,
    tiles_y: int,
    max_tile_px: int,
    include_full_frame: bool,
) -> list[VisionTile]:
    height, width, _ = image.shape
    tiles: list[VisionTile] = []
    index = 0
    if include_full_frame:
        full_bytes = _encode_png(_resize_max(image, max_tile_px))
        tiles.append(
            VisionTile(
                index=index,
                bbox_norm=(0.0, 0.0, 1.0, 1.0),
                image_bytes=full_bytes,
                kind="full",
            )
        )
        index += 1
    xs = [int(round(i * width / tiles_x)) for i in range(tiles_x + 1)]
    ys = [int(round(i * height / tiles_y)) for i in range(tiles_y + 1)]
    xs[-1] = width
    ys[-1] = height
    for row in range(tiles_y):
        for col in range(tiles_x):
            x0, x1 = xs[col], xs[col + 1]
            y0, y1 = ys[row], ys[row + 1]
            crop = image[y0:y1, x0:x1]
            crop_bytes = _encode_png(_resize_max(crop, max_tile_px))
            tiles.append(
                VisionTile(
                    index=index,
                    bbox_norm=(
                        x0 / width if width else 0.0,
                        y0 / height if height else 0.0,
                        x1 / width if width else 0.0,
                        y1 / height if height else 0.0,
                    ),
                    image_bytes=crop_bytes,
                    kind="tile",
                )
            )
            index += 1
    return tiles


def _resize_max(image: np.ndarray, max_px: int) -> np.ndarray:
    height, width, _ = image.shape
    longest = max(height, width)
    if longest <= max_px:
        return image
    scale = max_px / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    pil = Image.fromarray(image)
    resized = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return np.asarray(resized)


def _encode_png(image: np.ndarray) -> bytes:
    pil = Image.fromarray(image)
    buffer = BytesIO()
    pil.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
