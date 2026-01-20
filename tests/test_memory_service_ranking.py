import datetime as dt
from contextlib import contextmanager

import pytest

from autocapture.config import MemoryServiceConfig, MemoryServiceRetrievalConfig
from autocapture.memory_service.providers import HashEmbedder
from autocapture.memory_service.schemas import MemoryPolicyContext, MemoryQueryRequest
from autocapture.memory_service.store import CandidateItem, CandidateScores, MemoryServiceStore


class _DummyDialect:
    name = "sqlite"


class _DummyEngine:
    dialect = _DummyDialect()


class _DummyDB:
    engine = _DummyEngine()

    @contextmanager
    def session(self):
        yield None


class _DummyStore(MemoryServiceStore):
    def __init__(
        self,
        config: MemoryServiceConfig,
        items: list[CandidateItem],
        *,
        reranker=None,
    ) -> None:
        super().__init__(_DummyDB(), config, HashEmbedder(dim=config.embedder.dim), reranker)
        self._items = items

    def _load_memory_items(self, session, memory_ids: list[str]) -> list[CandidateItem]:
        return [item for item in self._items if item.memory_id in memory_ids]

    def _load_citations(self, memory_ids: list[str], *, namespace=None) -> dict[str, list]:
        _ = namespace
        return {memory_id: [] for memory_id in memory_ids}


def _candidate(
    memory_id: str,
    memory_type: str,
    text: str,
    *,
    created_at: dt.datetime,
) -> CandidateItem:
    return CandidateItem(
        memory_id=memory_id,
        memory_type=memory_type,
        content_text=text,
        content_json={},
        importance=0.5,
        trust_tier=0.5,
        created_at=created_at,
        sensitivity_rank=0,
        scores=CandidateScores(reasons={"semantic"}),
    )


def test_rank_candidates_deterministic_tiebreak() -> None:
    now = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    items = [
        _candidate("mem_b", "fact", "B", created_at=now),
        _candidate("mem_a", "fact", "A", created_at=now),
        _candidate("mem_c", "fact", "C", created_at=now),
    ]
    store = _DummyStore(MemoryServiceConfig(), items)
    candidates = {
        "mem_a": CandidateScores(semantic=1.0, keyword=1.0),
        "mem_b": CandidateScores(semantic=1.0, keyword=1.0),
        "mem_c": CandidateScores(semantic=1.0, keyword=1.0),
    }
    ranked = store._rank_candidates(candidates, now, "query")
    assert [item.memory_id for item in ranked] == ["mem_a", "mem_b", "mem_c"]


def test_pack_cards_type_priority_and_limits() -> None:
    now = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    retrieval = MemoryServiceRetrievalConfig(
        max_cards=10,
        max_tokens=200,
        max_per_type=1,
        type_priority=["decision", "procedure", "fact", "episodic", "glossary"],
    )
    config = MemoryServiceConfig(retrieval=retrieval)
    items = [
        _candidate("mem_fact", "fact", "Fact", created_at=now),
        _candidate("mem_decision", "decision", "Decision", created_at=now),
        _candidate("mem_proc", "procedure", "Procedure", created_at=now),
        _candidate("mem_fact2", "fact", "Fact2", created_at=now),
    ]
    for item in items:
        item.score = 1.0
    store = _DummyStore(config, items)
    request = MemoryQueryRequest(
        namespace="default",
        query="query",
        policy=MemoryPolicyContext(audience=["internal"], sensitivity_max="high"),
    )
    cards, truncated = store._pack_cards(items, request)
    assert [card.memory_id for card in cards] == ["mem_decision", "mem_proc", "mem_fact"]
    assert truncated is False


class _RecordingReranker:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.seen_texts: list[str] = []

    def score(self, _query: str, texts) -> list[float]:
        self.seen_texts = list(texts)
        return list(self._scores)[: len(self.seen_texts)]


def test_rerank_window_applies_to_top_candidates() -> None:
    now = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    retrieval = MemoryServiceRetrievalConfig(rerank_window=2)
    config = MemoryServiceConfig(enable_rerank=True, retrieval=retrieval)
    items = [
        _candidate("mem_a", "fact", "A", created_at=now),
        _candidate("mem_b", "fact", "B", created_at=now),
        _candidate("mem_c", "fact", "C", created_at=now),
    ]
    reranker = _RecordingReranker([0.0, 1.0])
    store = _DummyStore(config, items, reranker=reranker)
    candidates = {
        "mem_a": CandidateScores(semantic=1.0),
        "mem_b": CandidateScores(semantic=0.99),
        "mem_c": CandidateScores(semantic=0.1),
    }
    ranked = store._rank_candidates(candidates, now, "query")
    assert reranker.seen_texts == ["A", "B"]
    assert [item.memory_id for item in ranked] == ["mem_b", "mem_a", "mem_c"]


class _ExplodingSession:
    def execute(self, *_args, **_kwargs):
        raise AssertionError("DB should not be used for invalid policy")


class _ExplodingDialect:
    name = "postgresql"


class _ExplodingEngine:
    dialect = _ExplodingDialect()


class _ExplodingDB:
    engine = _ExplodingEngine()

    @contextmanager
    def session(self):
        yield _ExplodingSession()


def test_query_policy_rejects_before_db() -> None:
    config = MemoryServiceConfig()
    store = MemoryServiceStore(
        _ExplodingDB(), config, HashEmbedder(dim=config.embedder.dim), None
    )
    request = MemoryQueryRequest(
        namespace="default",
        query="query",
        policy=MemoryPolicyContext(audience=[], sensitivity_max="high"),
    )
    with pytest.raises(ValueError):
        store.query(request)
