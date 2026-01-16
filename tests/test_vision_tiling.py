from __future__ import annotations

import numpy as np
import pytest

from autocapture.vision.tiling import build_tiles


def test_tiling_order_is_deterministic() -> None:
    image = np.zeros((2160, 7680, 3), dtype=np.uint8)
    tiles = build_tiles(
        image,
        tiles_x=3,
        tiles_y=2,
        max_tile_px=1024,
        include_full_frame=True,
    )
    assert len(tiles) == 7
    assert tiles[0].kind == "full"
    assert tiles[0].bbox_norm == (0.0, 0.0, 1.0, 1.0)

    expected = [
        (0.0, 0.0, 1.0 / 3.0, 0.5),
        (1.0 / 3.0, 0.0, 2.0 / 3.0, 0.5),
        (2.0 / 3.0, 0.0, 1.0, 0.5),
        (0.0, 0.5, 1.0 / 3.0, 1.0),
        (1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0),
        (2.0 / 3.0, 0.5, 1.0, 1.0),
    ]
    for tile, exp in zip(tiles[1:], expected, strict=True):
        assert tile.kind == "tile"
        assert tile.bbox_norm == pytest.approx(exp)
