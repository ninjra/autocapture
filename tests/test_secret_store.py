import os

from autocapture.security.secret_store import SecretStore


def test_secret_store_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    store = SecretStore()
    record = store.get("OPENAI_API_KEY")
    assert record is not None
    assert record.key == "OPENAI_API_KEY"
    assert record.value == "sk-test-secret"


def test_secret_store_redacts_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    store = SecretStore()
    record = store.get("OPENAI_API_KEY")
    redacted = SecretStore.redact(record)
    assert redacted is not None
    assert "sk-test-secret" not in redacted
