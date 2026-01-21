from __future__ import annotations

from autocapture.ux.diff import diff_values


def test_diff_list_replaced() -> None:
    before = {"a": [1, 2], "b": {"c": 1}}
    after = {"a": [1, 3], "b": {"c": 1}}
    diff = diff_values(before, after)
    assert len(diff) == 1
    assert diff[0].path == "a"
    assert diff[0].before == [1, 2]
    assert diff[0].after == [1, 3]


def test_diff_nested_add_remove() -> None:
    before = {"a": {"b": 1}}
    after = {"a": {"b": 1, "c": 2}}
    diff = diff_values(before, after)
    assert any(item.path == "a.c" and item.kind == "add" for item in diff)
