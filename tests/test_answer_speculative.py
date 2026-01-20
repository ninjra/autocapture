from __future__ import annotations

import datetime as dt

import pytest

from autocapture.agents.answer_graph import AnswerGraph
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.memory.context_pack import EvidenceItem
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord
from autocapture.memory.prompts import PromptRegistry
from autocapture.memory.entities import EntityResolver, SecretStore
from autocapture.security.token_vault import TokenVaultStore
from autocapture.memory.retrieval import RetrievalService


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.stage: str | None = None

    async def generate_answer(
        self, system_prompt: str, query: str, context_pack_text: str, *, temperature=None
    ):
        _ = system_prompt, query, context_pack_text, temperature
        self.calls += 1
        if self.stage == "final_answer":
            return (
                "```json\n"
                '{"schema_version":2,"claims":[{"text":"Final answer","citations":[{"evidence_id":"E1","line_start":1,"line_end":1}]}]}'
                "\n```"
            )
        return "Draft answer [E1]"


@pytest.mark.anyio
async def test_speculative_early_exit(monkeypatch, tmp_path) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.capture.data_dir = tmp_path
    config.embed.text_model = "local-test"
    config.retrieval.speculative_enabled = True
    config.model_stages.draft_generate.enabled = True
    config.model_stages.final_answer.enabled = True

    db = DatabaseManager(config.database)
    retrieval = RetrievalService(db, config)
    prompt_registry = PromptRegistry.from_package("autocapture.prompts.derived")
    secret = SecretStore(tmp_path).get_or_create()
    entities = EntityResolver(db, secret, token_vault=TokenVaultStore(config, db))

    graph = AnswerGraph(config, retrieval, prompt_registry=prompt_registry, entities=entities)

    calls: list[str | None] = []
    event = EventRecord(
        event_id="event-1",
        ts_start=dt.datetime.now(dt.timezone.utc),
        ts_end=None,
        app_name="Docs",
        window_title="Notes",
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash",
        ocr_text="text",
        embedding_vector=None,
        tags={},
    )
    evidence = [
        EvidenceItem(
            evidence_id="E1",
            event_id="event-1",
            timestamp=event.ts_start.isoformat(),
            ts_end=None,
            app="Docs",
            title="Notes",
            domain=None,
            score=0.95,
            spans=[],
            text="text",
        )
    ]

    def fake_build_evidence(*args, **kwargs):
        calls.append(kwargs.get("retrieval_mode"))
        return evidence, [event], False

    graph._build_evidence = fake_build_evidence  # type: ignore[method-assign]

    fake_llm = FakeLLM()

    def _mock_select(self, stage: str, *, routing_override=None):
        _ = routing_override
        fake_llm.stage = stage
        return fake_llm, type("Decision", (), {"temperature": 0.2, "stage": stage})()

    monkeypatch.setattr("autocapture.model_ops.router.StageRouter.select_llm", _mock_select)

    result = await graph.run(
        "notes",
        time_range=None,
        filters=None,
        k=5,
        sanitized=False,
        extractive_only=False,
        routing={"llm": "ollama"},
    )
    assert result.answer.startswith("Final answer")
    assert calls == ["baseline", "deep"]
    assert fake_llm.calls == 3
