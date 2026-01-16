from __future__ import annotations

from tools.research_scout_workflow import compute_diff


def test_compute_diff_with_changes() -> None:
    old_report = {"ranked_items": [{"id": "a"}, {"id": "b"}]}
    new_report = {"ranked_items": [{"id": "a"}, {"id": "c"}]}

    diff = compute_diff(old_report, new_report, top_n=2)

    assert diff["changed"] == 1
    assert diff["ratio"] == 0.5


def test_compute_diff_without_old_report() -> None:
    new_report = {"ranked_items": [{"id": "a"}, {"id": "b"}]}

    diff = compute_diff(None, new_report, top_n=2)

    assert diff["changed"] == 2
    assert diff["ratio"] == 1.0
