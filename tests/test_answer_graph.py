from __future__ import annotations

import asyncio
import datetime as dt

from autocapture.agents.answer_graph import AnswerGraph
from autocapture.config import AppConfig, DatabaseConfig, ProviderRoutingConfig
from autocapture.memory.entities import EntityResolver, SecretStore
from autocapture.memory.retrieval import RetrievalService
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import CaptureRecord, EventRecord, OCRSpanRecord


class _StubEmbedder:
    model_name = "stub"
    dim = 3

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _StubVectorIndex:
    def search(self, *_args, **_kwargs):
        return []

    def upsert_spans(self, *_args, **_kwargs):
        return None


class _StubPromptRegistry:
    def get(self, _name):
        return type("Prompt", (), {"system_prompt": "Answer with citations"})()


class _StubProvider:
    async def generate_answer(self, *_args, **_kwargs):
        return "Answer [E1]"


class _RefineProvider:
    async def generate_answer(self, *_args, **_kwargs):
        return '{"refined_query": "hello notes"}'


class _BadRefineProvider:
    async def generate_answer(self, *_args, **_kwargs):
        return "not-json"


def _setup_graph() -> tuple[AnswerGraph, RetrievalService, DatabaseManager]:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    db = DatabaseManager(config.database)
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            CaptureRecord(
                id="evt-graph",
                captured_at=now,
                image_path=None,
                foreground_process="Editor",
                foreground_window="Notes",
                monitor_id="m1",
                is_fullscreen=False,
                ocr_status="done",
            )
        )
        session.flush()
        session.add(
            EventRecord(
                event_id="evt-graph",
                ts_start=now,
                ts_end=None,
                app_name="Editor",
                window_title="Notes",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="hello world",
                tags={},
            )
        )
        session.add(
            OCRSpanRecord(
                capture_id="evt-graph",
                span_key="S1",
                start=0,
                end=5,
                text="hello",
                confidence=0.9,
                bbox={"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            )
        )
    retrieval = RetrievalService(
        db, config, embedder=_StubEmbedder(), vector_index=_StubVectorIndex()
    )
    secret = SecretStore(config.capture.data_dir).get_or_create()
    entities = EntityResolver(db, secret)
    graph = AnswerGraph(config, retrieval, prompt_registry=_StubPromptRegistry(), entities=entities)
    return graph, retrieval, db


def test_answer_graph_extractive_only(monkeypatch) -> None:
    graph, _retrieval, _db = _setup_graph()
    result = asyncio.run(
        graph.run(
            "hello",
            time_range=None,
            filters=None,
            k=3,
            sanitized=False,
            extractive_only=True,
            routing=ProviderRoutingConfig().model_dump(),
            aggregates=None,
        )
    )
    assert result.answer
    assert result.citations


def test_answer_graph_filters_non_citable_results() -> None:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    config.embed.text_model = "local-test"
    db = DatabaseManager(config.database)
    now = dt.datetime.now(dt.timezone.utc)
    with db.session() as session:
        session.add(
            EventRecord(
                event_id="evt-nocite",
                ts_start=now,
                ts_end=None,
                app_name="Editor",
                window_title="Notes",
                url=None,
                domain=None,
                screenshot_path=None,
                screenshot_hash="hash",
                ocr_text="hello world",
                tags={},
            )
        )
    retrieval = RetrievalService(
        db, config, embedder=_StubEmbedder(), vector_index=_StubVectorIndex()
    )
    secret = SecretStore(config.capture.data_dir).get_or_create()
    entities = EntityResolver(db, secret)
    graph = AnswerGraph(config, retrieval, prompt_registry=_StubPromptRegistry(), entities=entities)

    evidence, _events, no_evidence = graph._build_evidence("hello", None, None, 2, sanitized=False)
    assert no_evidence is True
    assert evidence == []


def test_answer_graph_llm_path(monkeypatch) -> None:
    graph, _retrieval, _db = _setup_graph()

    def _select_llm(self, stage: str, *, routing_override=None):
        if stage == "query_refine":
            return _RefineProvider(), type("Decision", (), {"temperature": 0.2})()
        return _StubProvider(), type("Decision", (), {"temperature": 0.2})()

    monkeypatch.setattr("autocapture.model_ops.router.StageRouter.select_llm", _select_llm)
    result = asyncio.run(
        graph.run(
            "hello",
            time_range=None,
            filters=None,
            k=3,
            sanitized=False,
            extractive_only=False,
            routing=ProviderRoutingConfig().model_dump(),
            aggregates=None,
        )
    )
    assert result.answer
    assert "E1" in result.citations


def test_refine_query_uses_prompt_output(monkeypatch) -> None:
    graph, _retrieval, _db = _setup_graph()

    def _select_llm(self, stage: str, *, routing_override=None):
        return _RefineProvider(), type("Decision", (), {"temperature": 0.2})()

    monkeypatch.setattr("autocapture.model_ops.router.StageRouter.select_llm", _select_llm)
    evidence, _events, _no_evidence = graph._build_evidence("hello", None, None, 2, sanitized=False)
    refined = asyncio.run(graph._refine_query("hello", evidence, routing_override=None))
    assert refined.refined_query == "hello notes"


def test_refine_query_falls_back_on_invalid_output(monkeypatch) -> None:
    graph, _retrieval, _db = _setup_graph()

    def _select_llm(self, stage: str, *, routing_override=None):
        return _BadRefineProvider(), type("Decision", (), {"temperature": 0.2})()

    monkeypatch.setattr("autocapture.model_ops.router.StageRouter.select_llm", _select_llm)
    evidence, _events, _no_evidence = graph._build_evidence("hello", None, None, 2, sanitized=False)
    refined = asyncio.run(graph._refine_query("hello", evidence, routing_override=None))
    assert refined.refined_query == "hello Editor"


def test_answer_graph_no_evidence_returns_notice() -> None:
    config = AppConfig(database=DatabaseConfig(url="sqlite:///:memory:", sqlite_wal=False))
    config.features.enable_thresholding = True
    config.retrieval.lexical_min_score = 0.99
    db = DatabaseManager(config.database)
    retrieval = RetrievalService(
        db, config, embedder=_StubEmbedder(), vector_index=_StubVectorIndex()
    )
    secret = SecretStore(config.capture.data_dir).get_or_create()
    entities = EntityResolver(db, secret)
    graph = AnswerGraph(config, retrieval, prompt_registry=_StubPromptRegistry(), entities=entities)
    result = asyncio.run(
        graph.run(
            "nothing",
            time_range=None,
            filters=None,
            k=3,
            sanitized=False,
            extractive_only=True,
            routing=ProviderRoutingConfig().model_dump(),
            aggregates=None,
        )
    )
    assert result.answer
    assert "No evidence" in result.answer
