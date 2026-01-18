"""Citation overlay rendering helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

from PIL import Image, ImageDraw


def render_citation_overlay(
    image: Image.Image,
    bboxes: Iterable[Sequence[float]],
    *,
    normalized: bool | None = None,
    outline: tuple[int, int, int] = (255, 80, 80),
    fill_alpha: int = 64,
    line_width: int = 3,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = base.size
    for raw in bboxes:
        rect = _bbox_to_rect(raw, width, height, normalized=normalized)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle(
            [x0, y0, x1, y1],
            outline=outline + (255,),
            width=max(1, int(line_width)),
            fill=outline + (int(max(0, min(255, fill_alpha))),),
        )
    composed = Image.alpha_composite(base, overlay)
    return composed.convert("RGB")


def _bbox_to_rect(
    bbox: Sequence[float],
    width: int,
    height: int,
    *,
    normalized: bool | None,
) -> tuple[int, int, int, int] | None:
    values = [val for val in bbox if isinstance(val, (int, float))]
    if len(values) < 4:
        return None
    use_norm = normalized
    if use_norm is None:
        use_norm = max(values) <= 1.0
    if len(values) >= 8:
        xs = values[0::2]
        ys = values[1::2]
        if not xs or not ys:
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
    else:
        x0, y0, x1, y1 = values[:4]
    if use_norm:
        x0 *= width
        x1 *= width
        y0 *= height
        y1 *= height
    x0_i = int(max(0, min(width, round(x0))))
    x1_i = int(max(0, min(width, round(x1))))
    y0_i = int(max(0, min(height, round(y0))))
    y1_i = int(max(0, min(height, round(y1))))
    if x0_i > x1_i:
        x0_i, x1_i = x1_i, x0_i
    if y0_i > y1_i:
        y0_i, y1_i = y1_i, y0_i
    return x0_i, y0_i, x1_i, y1_i
