from __future__ import annotations

import pytest

from autocapture.api.server import create_app
from autocapture.config import AppConfig, DatabaseConfig


class _StubBatch:
    def __init__(self):
        self.results = []
        self.no_evidence = True
        self.reason = "empty"


@pytest.mark.anyio
async def test_no_evidence_skips_llm(tmp_path, async_client_factory, monkeypatch) -> None:
    config = AppConfig(
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        database=DatabaseConfig(url="sqlite:///:memory:"),
        tracking={"enabled": False},
        embed={"text_model": "local-test"},
    )
    app = create_app(config)
    retrieval = app.state.container.retrieval
    monkeypatch.setattr(retrieval, "retrieve", lambda *args, **kwargs: _StubBatch())

    def _fail(*_args, **_kwargs):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(app.state.container.policy_envelope, "execute_stage", _fail)
    monkeypatch.setattr(app.state.container.answer_graph, "run", _fail)

    async with async_client_factory(app) as client:
        response = await client.post("/api/answer", json={"query": "test"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["no_evidence"] is True
    assert payload["used_llm"] is False
    assert payload["banner"]["level"] == "no_evidence"
    assert payload["answer"] == ""
