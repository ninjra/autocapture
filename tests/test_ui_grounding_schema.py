from __future__ import annotations

from autocapture.vision.ui_grounding import _parse_payload, _normalize_elements, UIElement


def test_ui_grounding_parse_valid() -> None:
    raw = """
    {
      "elements": [
        {
          "id": "U1",
          "role": "button",
          "label": "Save",
          "bbox_norm": [0.1, 0.2, 0.3, 0.4],
          "click_point_norm": [0.2, 0.3],
          "confidence": 0.9
        }
      ]
    }
    """
    payload, meta = _parse_payload(raw)
    assert payload is not None
    assert not meta["parse_failed"]
    assert payload.elements[0].role == "button"


def test_ui_grounding_parse_invalid() -> None:
    payload, meta = _parse_payload("not json")
    assert payload is None
    assert meta["parse_failed"]


def test_ui_grounding_stable_ids() -> None:
    elements = [
        UIElement(
            id="",
            role="text",
            label="Title",
            bbox_norm=[0.0, 0.0, 0.2, 0.1],
            confidence=0.8,
        )
    ]
    normalized = _normalize_elements(elements)
    assert normalized[0].id == "U1"
