from __future__ import annotations

from autocapture.vision.layout import build_layout


def test_layout_blocks_and_markdown() -> None:
    spans = [
        {"text": "DASHBOARD", "bbox": [0, 0, 120, 0, 120, 20, 0, 20]},
        {"text": "- One", "bbox": [0, 30, 80, 30, 80, 50, 0, 50]},
        {"text": "- Two", "bbox": [0, 60, 80, 60, 80, 80, 0, 80]},
        {"text": "ColA", "bbox": [0, 100, 60, 100, 60, 120, 0, 120]},
        {"text": "ColB", "bbox": [80, 100, 140, 100, 140, 120, 80, 120]},
        {"text": "1", "bbox": [0, 130, 60, 130, 60, 150, 0, 150]},
        {"text": "2", "bbox": [80, 130, 140, 130, 140, 150, 80, 150]},
    ]
    blocks, markdown = build_layout(spans)
    assert blocks
    assert "## DASHBOARD" in markdown
    assert "- One" in markdown
    assert "| ColA | ColB |" in markdown
    assert "| 1 | 2 |" in markdown
