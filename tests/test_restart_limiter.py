from __future__ import annotations

from autocapture.ui.restart_limiter import RestartLimiter


class _Time:
    def __init__(self, start: float = 0.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value


def test_restart_limiter_blocks_after_threshold() -> None:
    clock = _Time()
    limiter = RestartLimiter(window_s=60.0, max_attempts=2, cooldown_s=30.0, time_source=clock)

    assert limiter.attempt().allowed
    assert limiter.attempt().allowed
    blocked = limiter.attempt()
    assert not blocked.allowed
    assert blocked.reason == "loop"
    assert blocked.cooldown_remaining_s == 30.0

    clock.value += 10.0
    still_blocked = limiter.attempt()
    assert not still_blocked.allowed
    assert still_blocked.reason == "cooldown"
    assert still_blocked.cooldown_remaining_s == 20.0

    clock.value += 21.0
    assert limiter.attempt().allowed


def test_restart_limiter_resets_after_window() -> None:
    clock = _Time()
    limiter = RestartLimiter(window_s=5.0, max_attempts=2, cooldown_s=10.0, time_source=clock)

    assert limiter.attempt().allowed
    clock.value += 6.0
    assert limiter.attempt().allowed
    assert limiter.attempt().allowed
