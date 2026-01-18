from __future__ import annotations

from PIL import Image

from autocapture.vision.citation_overlay import render_citation_overlay


def test_citation_overlay_draws_bbox() -> None:
    image = Image.new("RGB", (64, 64), (0, 0, 0))
    overlay = render_citation_overlay(
        image,
        bboxes=[[8, 8, 24, 24]],
        normalized=False,
        fill_alpha=200,
        line_width=2,
    )
    assert overlay.size == image.size
    px = overlay.load()
    assert px is not None
    assert px[8, 8] != (0, 0, 0)
