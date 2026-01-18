from autocapture.observability.otel import (
    exported_spans,
    init_otel,
    otel_span,
    record_histogram,
    safe_attributes,
)


def test_otel_safe_attributes_filters_keys():
    attrs = safe_attributes(
        {
            "stage_name": "capture_frame",
            "query": "secret query",
            "provider_id": "local",
            "window_title": "secret",
        }
    )
    assert "stage_name" in attrs
    assert "provider_id" in attrs
    assert "query" not in attrs
    assert "window_title" not in attrs


def test_otel_span_records_without_sensitive_attrs():
    init_otel(True, test_mode=True)
    with otel_span("capture_frame", {"stage_name": "capture_frame", "query": "secret"}):
        record_histogram("capture_frame_ms", 12.0, {"stage_name": "capture_frame"})
    spans = list(exported_spans())
    assert spans
    attrs = spans[-1].attributes
    assert "stage_name" in attrs
    assert "query" not in attrs
