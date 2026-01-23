from __future__ import annotations

from autocapture.runtime_cpu import reduce_worker_counts


def test_reduce_worker_counts() -> None:
    desired = {"ocr": 4, "embed": 2, "agents": 1}
    reduced = reduce_worker_counts(desired)
    assert reduced["ocr"] == 1
    assert reduced["embed"] == 0
    assert reduced["agents"] == 0
