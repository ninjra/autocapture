from __future__ import annotations

from autocapture.vision.layout import build_layout


def test_layout_markdown_tie_breakers() -> None:
    spans = [
        {"text": "b", "bbox": [0, 0, 10, 0, 10, 10, 0, 10]},
        {"text": "a", "bbox": [0, 0, 10, 0, 10, 10, 0, 10]},
        {"text": "c", "bbox": [0, 20, 10, 20, 10, 30, 0, 30]},
    ]
    blocks, markdown = build_layout(spans)
    assert blocks
    lines = [
        line
        for line in markdown.splitlines()
        if line.strip() and not line.strip().startswith("| ---")
    ]
    assert lines[0].strip() == "| b | a |"
    assert lines[1].strip() == "c"


def test_layout_block_order_is_deterministic() -> None:
    spans = [
        {"text": "z", "bbox": [5, 10, 15, 10, 15, 20, 5, 20]},
        {"text": "a", "bbox": [5, 0, 15, 0, 15, 10, 5, 10]},
        {"text": "b", "bbox": [25, 0, 35, 0, 35, 10, 25, 10]},
    ]
    blocks, markdown = build_layout(spans)
    assert blocks
    lines = [
        line
        for line in markdown.splitlines()
        if line.strip() and not line.strip().startswith("| ---")
    ]
    assert lines[0].strip() == "| a | b |"
    assert lines[1].strip() == "z"
