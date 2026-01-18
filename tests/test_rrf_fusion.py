from __future__ import annotations

import datetime as dt

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.memory.retrieval import RetrievalResult, RetrievalService, _rrf_fuse
from autocapture.storage.database import DatabaseManager
from autocapture.storage.models import EventRecord


def _event(event_id: str) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        ts_start=dt.datetime.now(dt.timezone.utc),
        ts_end=None,
        app_name="App",
        window_title="Title",
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash",
        ocr_text="text",
        tags={},
    )


def test_rrf_fusion_tie_breakers() -> None:
    event_a = _event("event-a")
    event_b = _event("event-b")
    list_a = [
        RetrievalResult(event=event_a, score=1.0, matched_span_keys=["S1"]),
        RetrievalResult(event=event_b, score=0.9, matched_span_keys=["S2"]),
    ]
    list_b = [
        RetrievalResult(event=event_b, score=1.0, matched_span_keys=["S3"]),
        RetrievalResult(event=event_a, score=0.9, matched_span_keys=["S4"]),
    ]
    fused = _rrf_fuse([list_a, list_b], 60)
    assert fused[0].event.event_id == "event-a"
    assert fused[1].event.event_id == "event-b"
    assert fused[0].matched_span_keys == ["S1", "S4"]
    assert fused[1].matched_span_keys == ["S2", "S3"]


def test_rewrite_gating_skips_when_confident(monkeypatch, tmp_path) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.embed.text_model = "local-test"
    config.retrieval.fusion_enabled = True
    config.retrieval.multi_query_enabled = True
    config.retrieval.rrf_enabled = True
    config.retrieval.fusion_confidence_min = 0.5
    config.retrieval.fusion_rank_gap_min = 0.1
    db = DatabaseManager(config.database)
    service = RetrievalService(db, config)
    event = _event("event-1")

    def _fake_candidates(*_args, **_kwargs):
        return [RetrievalResult(event=event, score=0.9)]

    called = {"rewrites": 0}

    def _fake_rewrite(_query: str):
        called["rewrites"] += 1
        return ["rewrite"]

    monkeypatch.setattr(service, "_retrieve_candidates", _fake_candidates)
    monkeypatch.setattr(service, "_rewrite_queries", _fake_rewrite)
    service.retrieve("query", None, None, limit=2)
    assert called["rewrites"] == 0


def test_rerank_deterministic_ordering(monkeypatch, tmp_path) -> None:
    class StubReranker:
        def rank(self, _query: str, _documents, **_kwargs):
            return [0.1, 0.9]

    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.embed.text_model = "local-test"
    config.routing.reranker = "enabled"
    config.reranker.enabled = True
    db = DatabaseManager(config.database)
    service = RetrievalService(db, config, reranker=StubReranker())
    event_a = _event("event-a")
    event_b = _event("event-b")
    results = [
        RetrievalResult(event=event_a, score=0.8),
        RetrievalResult(event=event_b, score=0.7),
    ]
    reranked = service._rerank_results("query", results)
    assert reranked[0].event.event_id == "event-b"
