from __future__ import annotations

import datetime as dt

from autocapture.memory.retrieval import _assign_ranks, _rrf_fuse, RetrievedEvent
from autocapture.storage.models import EventRecord


def _event(event_id: str) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        ts_start=dt.datetime.now(dt.timezone.utc),
        ts_end=None,
        app_name="App",
        window_title="Window",
        url=None,
        domain=None,
        screenshot_path=None,
        screenshot_hash="hash",
        ocr_text="text",
        embedding_vector=None,
        tags={},
    )


def test_rrf_fusion_prefers_consistent_hit() -> None:
    e1 = _event("E1")
    e2 = _event("E2")
    list_a = [
        RetrievedEvent(event=e1, score=0.9),
        RetrievedEvent(event=e2, score=0.2),
    ]
    list_b = [RetrievedEvent(event=e1, score=0.8)]
    fused = _rrf_fuse([list_a, list_b], rrf_k=60)
    assert fused
    assert fused[0].event.event_id == "E1"


def test_assign_ranks_sets_gap() -> None:
    e1 = _event("E1")
    e2 = _event("E2")
    ranked = _assign_ranks(
        [
            RetrievedEvent(event=e1, score=1.0),
            RetrievedEvent(event=e2, score=0.7),
        ]
    )
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
    assert ranked[1].rank_gap == 0.3
