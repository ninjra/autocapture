"""Clock helpers for deterministic overlay tracker logic."""

from __future__ import annotations

import datetime as dt
from typing import Protocol


class Clock(Protocol):
    def now(self) -> dt.datetime: ...


class SystemClock:
    def now(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)


class FixedClock:
    def __init__(self, now: dt.datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("FixedClock requires timezone-aware datetime")
        self._now = now

    def now(self) -> dt.datetime:
        return self._now

    def set(self, now: dt.datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("FixedClock requires timezone-aware datetime")
        self._now = now
