"""Time helpers for monotonic-safe latency calculations."""

from __future__ import annotations

import datetime as dt
import time


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def monotonic_now() -> float:
    return time.monotonic()


def elapsed_ms(start_ts: float, end_ts: float | None = None) -> float:
    end_ts = monotonic_now() if end_ts is None else end_ts
    return max(0.0, (end_ts - start_ts) * 1000)
