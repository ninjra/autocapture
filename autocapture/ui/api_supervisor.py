"""API supervision helper for tray resiliency."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class SupervisorState:
    status: str
    attempts: int
    next_restart_at: float


class ApiSupervisor:
    def __init__(
        self,
        health_check: Callable[[], bool],
        restart: Callable[[], None],
        *,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._health_check = health_check
        self._restart = restart
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._time_source = time_source
        self._attempts = 0
        self._next_restart_at = 0.0
        self._status = "unknown"

    def tick(self) -> SupervisorState:
        now = self._time_source()
        healthy = self._health_check()
        if healthy:
            self._attempts = 0
            self._next_restart_at = 0.0
            self._status = "healthy"
            return self._state()

        if now < self._next_restart_at:
            self._status = "backoff"
            return self._state()

        self._attempts += 1
        self._status = "restarting"
        self._restart()
        delay = min(self._backoff_base_s * (2 ** (self._attempts - 1)), self._backoff_max_s)
        self._next_restart_at = now + delay
        return self._state()

    def _state(self) -> SupervisorState:
        return SupervisorState(
            status=self._status,
            attempts=self._attempts,
            next_restart_at=self._next_restart_at,
        )
