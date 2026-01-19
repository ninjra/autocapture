from __future__ import annotations

import datetime as dt

import pytest

from autocapture.agents.answer_graph import AnswerGraph
from autocapture.config import AppConfig, DatabaseConfig
from autocapture.memory.context_pack import EvidenceItem
from autocapture.memory.entities import EntityResolver, SecretStore
from autocapture.memory.prompts import PromptRegistry
from autocapture.memory.retrieval import RetrievalService
from autocapture.security.token_vault import TokenVaultStore
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord


class StubLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.stage: str | None = None

    async def generate_answer(self, system_prompt, query, context_pack_text, *, temperature=None):
        _ = system_prompt, query, context_pack_text, temperature
        self.calls.append(query)
        if self.stage == "final_answer":
            return (
                "```json\n"
                '{"schema_version":2,"claims":[{"text":"Final answer","citations":[{"evidence_id":"E1","line_start":1,"line_end":1}]}]}'
                "\n```"
            )
        return "Draft answer [E1]"


def _event(event_id: str) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        ts_start=dt.datetime.now(dt.timezone.utc),
        ts_end=None,
        app_name="Docs",
        window_title="Notes",
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash",
        ocr_text="text",
        tags={},
    )


def _evidence(event: EventRecord, score: float, evidence_id: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        event_id=event.event_id,
        timestamp=event.ts_start.isoformat(),
        ts_end=None,
        app="Docs",
        title="Notes",
        domain=None,
        score=score,
        spans=[],
        text="text",
    )


@pytest.mark.anyio
async def test_speculative_verifier_blocks_early_exit(monkeypatch, tmp_path) -> None:
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

    event = _event("event-1")
    evidence = [_evidence(event, 0.9, "E1")]

    def fake_build_evidence(*_args, **_kwargs):
        return evidence, [event], False

    graph._build_evidence = fake_build_evidence  # type: ignore[method-assign]

    verifier_calls = []

    def fake_verify(_answer, _citations, _evidence, _verifier=None):
        verifier_calls.append(True)
        return False

    monkeypatch.setattr("autocapture.agents.answer_graph._verify_answer", fake_verify)

    stub = StubLLM()

    def _mock_select(self, stage: str, *, routing_override=None):
        _ = routing_override
        stub.stage = stage
        return stub, type("Decision", (), {"temperature": 0.2, "stage": stage, "cloud": False})()

    monkeypatch.setattr("autocapture.model_ops.router.StageRouter.select_llm", _mock_select)

    await graph.run(
        "notes",
        time_range=None,
        filters=None,
        k=5,
        sanitized=False,
        extractive_only=False,
        routing={"llm": "ollama"},
    )
    assert len(verifier_calls) >= 1
    assert len(stub.calls) >= 2


@pytest.mark.anyio
async def test_speculative_low_confidence_forces_deep(monkeypatch, tmp_path) -> None:
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

    event = _event("event-1")
    evidence = [_evidence(event, 0.4, "E1"), _evidence(event, 0.4, "E2")]

    def fake_build_evidence(*_args, **_kwargs):
        return evidence, [event], False

    graph._build_evidence = fake_build_evidence  # type: ignore[method-assign]
    monkeypatch.setattr(
        "autocapture.agents.answer_graph._verify_answer", lambda *_args, **_kwargs: True
    )

    stub = StubLLM()

    def _mock_select(self, stage: str, *, routing_override=None):
        _ = routing_override
        stub.stage = stage
        return stub, type("Decision", (), {"temperature": 0.2, "stage": stage, "cloud": False})()

    monkeypatch.setattr("autocapture.model_ops.router.StageRouter.select_llm", _mock_select)

    await graph.run(
        "notes",
        time_range=None,
        filters=None,
        k=5,
        sanitized=False,
        extractive_only=False,
        routing={"llm": "ollama"},
    )
    assert len(stub.calls) >= 2


@pytest.mark.anyio
async def test_speculative_verified_early_exit(monkeypatch, tmp_path) -> None:
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

    event = _event("event-1")
    evidence = [_evidence(event, 0.9, "E1")]

    def fake_build_evidence(*_args, **_kwargs):
        return evidence, [event], False

    graph._build_evidence = fake_build_evidence  # type: ignore[method-assign]
    monkeypatch.setattr(
        "autocapture.agents.answer_graph._verify_answer", lambda *_args, **_kwargs: True
    )

    stub = StubLLM()

    def _mock_select(self, stage: str, *, routing_override=None):
        _ = routing_override
        stub.stage = stage
        return stub, type("Decision", (), {"temperature": 0.2, "stage": stage, "cloud": False})()

    monkeypatch.setattr("autocapture.model_ops.router.StageRouter.select_llm", _mock_select)

    await graph.run(
        "notes",
        time_range=None,
        filters=None,
        k=5,
        sanitized=False,
        extractive_only=False,
        routing={"llm": "ollama"},
    )
    assert len(stub.calls) >= 2
