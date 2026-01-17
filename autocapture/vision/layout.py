"""Deterministic OCR layout reconstruction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class _Line:
    text: str
    spans: list[dict]
    bbox: tuple[int, int, int, int]


def build_layout(spans: list[dict]) -> tuple[list[dict], str]:
    lines = _cluster_lines(spans)
    blocks: list[dict] = []
    md_lines: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if _is_table_line(line):
            group, idx = _consume_table(lines, idx)
            block = _table_block(group)
            blocks.append(block)
            md_lines.extend(block["markdown"].splitlines())
            md_lines.append("")
            continue
        if _is_list_line(line.text):
            group, idx = _consume_list(lines, idx)
            block = _list_block(group)
            blocks.append(block)
            md_lines.extend(block["markdown"].splitlines())
            md_lines.append("")
            continue
        block = _line_block(line)
        blocks.append(block)
        md_lines.append(block["markdown"])
        md_lines.append("")
        idx += 1
    layout_md = "\n".join(_trim_blank_lines(md_lines)).strip()
    return blocks, layout_md


def _cluster_lines(spans: list[dict]) -> list[_Line]:
    items: list[tuple[dict, tuple[int, int, int, int], int]] = []
    heights: list[int] = []
    for span in spans or []:
        bbox = _span_bbox(span)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        height = max(1, y1 - y0)
        heights.append(height)
        items.append((span, bbox, height))
    if not items:
        return []
    threshold = max(6, int(median(heights) * 0.7))
    items.sort(key=lambda item: (item[1][1], item[1][0]))
    lines: list[list[tuple[dict, tuple[int, int, int, int]]]] = []
    line_centers: list[int] = []
    for span, bbox, _height in items:
        y_center = (bbox[1] + bbox[3]) // 2
        placed = False
        for idx, center in enumerate(line_centers):
            if abs(y_center - center) <= threshold:
                lines[idx].append((span, bbox))
                line_centers[idx] = int((line_centers[idx] + y_center) / 2)
                placed = True
                break
        if not placed:
            lines.append([(span, bbox)])
            line_centers.append(y_center)

    result: list[_Line] = []
    for group in lines:
        group.sort(key=lambda item: item[1][0])
        texts = [str(item[0].get("text") or "").strip() for item in group]
        joined = " ".join([text for text in texts if text])
        bbox = _merge_bbox([item[1] for item in group])
        result.append(_Line(text=joined, spans=[item[0] for item in group], bbox=bbox))
    result.sort(key=lambda line: (line.bbox[1], line.bbox[0]))
    return result


def _span_bbox(span: dict) -> tuple[int, int, int, int] | None:
    raw = span.get("bbox")
    if isinstance(raw, dict):
        coords = [raw.get(key) for key in ("x0", "y0", "x1", "y1")]
    else:
        coords = list(raw or [])
    if len(coords) >= 8:
        xs = coords[0::2]
        ys = coords[1::2]
        if not xs or not ys:
            return None
        x0, x1 = int(min(xs)), int(max(xs))
        y0, y1 = int(min(ys)), int(max(ys))
        return x0, y0, x1, y1
    if len(coords) >= 4:
        x0, y0, x1, y1 = [int(val) for val in coords[:4]]
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    return None


def _merge_bbox(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    xs0 = [bbox[0] for bbox in bboxes]
    ys0 = [bbox[1] for bbox in bboxes]
    xs1 = [bbox[2] for bbox in bboxes]
    ys1 = [bbox[3] for bbox in bboxes]
    return min(xs0), min(ys0), max(xs1), max(ys1)


def _is_table_line(line: _Line) -> bool:
    return len(line.spans) >= 2


def _consume_table(lines: list[_Line], start: int) -> tuple[list[_Line], int]:
    group: list[_Line] = []
    idx = start
    while idx < len(lines) and _is_table_line(lines[idx]):
        group.append(lines[idx])
        idx += 1
    if len(group) < 2:
        return [lines[start]], start + 1
    return group, idx


def _table_block(lines: list[_Line]) -> dict:
    rows = []
    max_cols = max(len(line.spans) for line in lines)
    for line in lines:
        parts = [str(span.get("text") or "").strip() for span in line.spans]
        parts = [part for part in parts if part]
        if len(parts) < max_cols:
            parts.extend([""] * (max_cols - len(parts)))
        rows.append(parts[:max_cols])
    header = rows[0]
    divider = ["---"] * max_cols
    markdown_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(divider) + " |"]
    for row in rows[1:]:
        markdown_lines.append("| " + " | ".join(row) + " |")
    bbox = _merge_bbox([line.bbox for line in lines])
    return {
        "type": "table",
        "rows": rows,
        "bbox": list(bbox),
        "markdown": "\n".join(markdown_lines),
    }


def _is_list_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] in {"-", "*", "•"}:
        return True
    if stripped[:2].isdigit() and stripped[2:3] in {".", ")"}:
        return True
    return False


def _consume_list(lines: list[_Line], start: int) -> tuple[list[_Line], int]:
    group: list[_Line] = []
    idx = start
    while idx < len(lines) and _is_list_line(lines[idx].text):
        group.append(lines[idx])
        idx += 1
    return group, idx


def _list_block(lines: list[_Line]) -> dict:
    items: list[str] = []
    for line in lines:
        text = line.text.strip()
        if text.startswith(("-", "*", "•")):
            text = text[1:].strip()
        items.append(text)
    markdown_lines = [f"- {item}" for item in items if item]
    bbox = _merge_bbox([line.bbox for line in lines])
    return {
        "type": "list",
        "items": items,
        "bbox": list(bbox),
        "markdown": "\n".join(markdown_lines),
    }


def _line_block(line: _Line) -> dict:
    text = line.text.strip()
    heading = _is_heading(text)
    prefix = "## " if heading else ""
    return {
        "type": "heading" if heading else "line",
        "text": text,
        "bbox": list(line.bbox),
        "markdown": f"{prefix}{text}",
    }


def _is_heading(text: str) -> bool:
    if not text:
        return False
    if len(text) <= 50 and text.isupper():
        return True
    if len(text) <= 40 and text.endswith(":"):
        return True
    return False


def _trim_blank_lines(lines: list[str]) -> list[str]:
    trimmed: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if blank:
                continue
            blank = True
            trimmed.append("")
        else:
            blank = False
            trimmed.append(line)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return trimmed
