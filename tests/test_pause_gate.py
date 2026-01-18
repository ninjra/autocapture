from __future__ import annotations

import pytest

from autocapture.runtime_pause import PauseController


def test_pause_gate_file_semantics(tmp_path) -> None:
    latch = tmp_path / "pause.flag"
    reason = tmp_path / "pause_reason.json"
    pause = PauseController(latch, reason, poll_interval_s=0.01)

    pause.pause("manual", "unit")
    assert latch.exists()
    state = pause.get_state()
    assert state.is_paused is True
    assert state.reason == "manual"

    pause.pause({"reason": "fullscreen", "app": "demo"}, "unit")
    state = pause.get_state()
    assert state.reason == "fullscreen"

    pause.resume("unit")
    assert not latch.exists()
    state = pause.get_state()
    assert state.is_paused is False


def test_pause_wait_timeout(tmp_path) -> None:
    latch = tmp_path / "pause.flag"
    reason = tmp_path / "pause_reason.json"
    pause = PauseController(latch, reason, poll_interval_s=0.01)

    pause.pause("manual", "unit")
    with pytest.raises(TimeoutError):
        pause.wait_until_resumed(timeout=0.05)

    pause.resume("unit")
    assert pause.wait_until_resumed(timeout=0.05) is True


def test_pause_reason_corrupt(tmp_path) -> None:
    latch = tmp_path / "pause.flag"
    reason = tmp_path / "pause_reason.json"
    pause = PauseController(latch, reason, poll_interval_s=0.01)

    pause.pause("manual", "unit")
    reason.write_text("{bad json", encoding="utf-8")
    state = pause.get_state()
    assert state.is_paused is True
    assert state.reason == "unknown"
    pause.resume("unit")
