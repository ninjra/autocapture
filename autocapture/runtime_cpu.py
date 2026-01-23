"""CPU usage sampling and throttling helpers."""

from __future__ import annotations

import time
from typing import Callable


class CpuUsageSampler:
    def __init__(
        self,
        *,
        sample_interval_s: float = 2.0,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sample_interval_s = max(0.1, sample_interval_s)
        self._time_source = time_source
        self._process = None
        self._cpu_count = 1
        self._primed = False
        self._last_sample_at = 0.0
        self._last_value: float | None = None
        try:
            import psutil  # type: ignore

            self._process = psutil.Process()
            self._cpu_count = psutil.cpu_count() or 1
        except Exception:
            self._process = None

    def sample(self) -> float | None:
        if self._process is None:
            return None
        now = self._time_source()
        if not self._primed:
            self._process.cpu_percent(interval=None)
            self._primed = True
            self._last_sample_at = now
            return None
        if self._last_value is not None and now - self._last_sample_at < self._sample_interval_s:
            return self._last_value
        raw = float(self._process.cpu_percent(interval=None))
        self._last_sample_at = now
        normalized = raw / max(1, self._cpu_count)
        normalized = max(0.0, min(normalized, 100.0))
        self._last_value = normalized
        return normalized


def reduce_worker_counts(desired: dict[str, int]) -> dict[str, int]:
    reduced = dict(desired)
    if reduced.get("agents", 0) > 0:
        reduced["agents"] = 0
    if reduced.get("embed", 0) > 0:
        reduced["embed"] = 0
    if reduced.get("ocr", 0) > 1:
        reduced["ocr"] = 1
    return reduced
