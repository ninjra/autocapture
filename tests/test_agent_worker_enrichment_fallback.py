from autocapture.worker import agent_worker


def test_fallback_enrichment_payload_fills_required_fields() -> None:
    context = {
        "event_id": "evt-123",
        "app_name": "TestApp",
        "window_title": "Test Window",
        "ocr_text": "Reviewing ERP records and updating notes.",
    }
    raw = (
        '{"schema_version":"2.0","short_summary":123,'
        '"what_i_was_doing":["a","b"],"provenance":[]}'
    )
    result = agent_worker._fallback_enrichment_payload(
        raw,
        context,
        response_model="model-x",
        response_provider="provider-y",
        prompt_id="event_enrichment:v1",
    )
    assert result.event_id == "evt-123"
    assert result.short_summary
    assert result.what_i_was_doing
    assert 0.0 <= result.importance <= 1.0
    assert result.provenance.model == "model-x"
    assert result.provenance.provider == "provider-y"
