from autocapture.vision.types import build_ocr_payload
from autocapture.worker.event_worker import _clamp_bbox


def test_ocr_span_offsets_align_to_raw_text():
    text, spans = build_ocr_payload(
        [
            ("Alpha", 0.9, [0, 0, 10, 10]),
            ("Beta", 0.8, [0, 12, 10, 22]),
        ]
    )
    for span in spans:
        start = span["start"]
        end = span["end"]
        assert text[start:end] == span["text"]


def test_ocr_bbox_clamped_to_bounds():
    bbox = [-5, -10, 120, 130]
    bounds = [0, 0, 100, 100]
    clamped = _clamp_bbox(bbox, bounds)
    assert isinstance(clamped, list)
    assert min(clamped) >= 0
    assert clamped[0] <= 100
    assert clamped[1] <= 100
