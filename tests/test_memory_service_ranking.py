import datetime as dt
from contextlib import contextmanager

from autocapture.config import MemoryServiceConfig, MemoryServiceRetrievalConfig
from autocapture.memory_service.providers import HashEmbedder
from autocapture.memory_service.schemas import MemoryPolicyContext, MemoryQueryRequest
from autocapture.memory_service.store import CandidateItem, CandidateScores, MemoryServiceStore


class _DummyDB:
    @contextmanager
    def session(self):
        yield None


class _DummyStore(MemoryServiceStore):
    def __init__(self, config: MemoryServiceConfig, items: list[CandidateItem]) -> None:
        super().__init__(_DummyDB(), config, HashEmbedder(dim=config.embedder.dim), None)
        self._items = items

    def _load_memory_items(self, session, memory_ids: list[str]) -> list[CandidateItem]:
        return [item for item in self._items if item.memory_id in memory_ids]

    def _load_citations(self, memory_ids: list[str]) -> dict[str, list]:
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
