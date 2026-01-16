from __future__ import annotations

from autocapture.vision.extractors import _parse_payload


def test_parse_payload_json() -> None:
    raw = """
    {
      "screen_summary": "summary",
      "visible_text": "alpha beta",
      "regions": [
        {
          "bbox_norm": [0.0, 0.0, 1.0, 1.0],
          "label": "window",
          "app_hint": "app",
          "title_hint": "title",
          "text_verbatim": "alpha",
          "keywords": ["alpha"],
          "confidence": 0.9
        }
      ],
      "tables_detected": false,
      "spreadsheets_detected": false
    }
    """
    payload, meta = _parse_payload(raw)
    assert payload is not None
    assert meta.parse_failed is False
    assert payload.regions[0].text_verbatim == "alpha"


def test_parse_payload_tron() -> None:
    raw = (
        "class Region { bbox_norm label app_hint title_hint text_verbatim keywords confidence }\n"
        "class Extract { screen_summary visible_text regions tables_detected spreadsheets_detected }\n\n"
        'Extract("summary", "alpha beta", [Region([0,0,1,1], "window", "app", "title", "alpha", '
        '["alpha"], 0.9)], false, false)'
    )
    payload, meta = _parse_payload(raw)
    assert payload is not None
    assert meta.parse_failed is False
    assert payload.visible_text == "alpha beta"
