from __future__ import annotations

from autocapture.ux.redaction import redact_payload


def test_redaction_masks_secret_fields() -> None:
    payload = {
        "api_key": "sk-secret",
        "token": "abc123",
        "nested": {"secret": "value"},
    }
    redacted = redact_payload(payload)
    assert redacted["api_key"] != "sk-secret"
    assert redacted["token"] != "abc123"
    assert redacted["nested"]["secret"] != "value"
