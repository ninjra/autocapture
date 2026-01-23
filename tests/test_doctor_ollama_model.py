from __future__ import annotations

from types import SimpleNamespace

import autocapture.doctor as doctor


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


def test_ollama_model_available_accepts_latest_tag(monkeypatch) -> None:
    payload = {"models": [{"name": "llama3:latest"}, {"name": "llava:latest"}]}

    def _fake_get(_url, headers=None):
        return _FakeResponse(200, payload), None

    monkeypatch.setattr(doctor, "_http_get", _fake_get)
    result = doctor._check_ollama_endpoint(
        name="llm", base_url="http://127.0.0.1:11434", model="llama3"
    )
    assert result.ok


def test_ollama_model_missing_with_tag(monkeypatch) -> None:
    payload = {"models": [{"name": "llama3:latest"}]}

    def _fake_get(_url, headers=None):
        return _FakeResponse(200, payload), None

    monkeypatch.setattr(doctor, "_http_get", _fake_get)
    result = doctor._check_ollama_endpoint(
        name="llm", base_url="http://127.0.0.1:11434", model="llama3:instruct"
    )
    assert not result.ok
