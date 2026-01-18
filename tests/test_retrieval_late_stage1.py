from __future__ import annotations

import datetime as dt

from autocapture.memory import retrieval as retrieval_module


def test_late_stage1_window_ok() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    assert retrieval_module._late_stage1_window_ok(  # type: ignore[attr-defined]
        (now - dt.timedelta(days=1), now), 7
    )
    assert not retrieval_module._late_stage1_window_ok(  # type: ignore[attr-defined]
        (now - dt.timedelta(days=10), now), 7
    )
