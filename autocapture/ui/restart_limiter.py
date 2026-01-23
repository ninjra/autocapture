"""API restart guard to prevent restart loops."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class RestartDecision:
    allowed: bool
    cooldown_remaining_s: float
    reason: str | None


class RestartLimiter:
    def __init__(
        self,
        *,
        window_s: float = 60.0,
        max_attempts: int = 3,
        cooldown_s: float = 120.0,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window_s = window_s
        self._max_attempts = max_attempts
        self._cooldown_s = cooldown_s
        self._time_source = time_source
        self._history: deque[float] = deque()
        self._blocked_until = 0.0

    def attempt(self) -> RestartDecision:
        now = self._time_source()
        if now < self._blocked_until:
            return RestartDecision(False, self._blocked_until - now, "cooldown")
        while self._history and now - self._history[0] > self._window_s:
            self._history.popleft()
        if len(self._history) >= self._max_attempts:
            self._blocked_until = now + self._cooldown_s
            self._history.clear()
            return RestartDecision(False, self._cooldown_s, "loop")
        self._history.append(now)
        return RestartDecision(True, 0.0, None)
