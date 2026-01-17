"""Global LLM concurrency governor with priority and adaptive limits."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import psutil

from ..observability.metrics import get_gpu_snapshot


@dataclass(frozen=True)
class PressureSample:
    cpu_util: float | None
    gpu_util: float | None


class PressureProvider(Protocol):
    def sample(self) -> PressureSample: ...


class SystemPressureProvider:
    def __init__(self) -> None:
        self._process = psutil.Process()

    def sample(self) -> PressureSample:
        cpu_percent = self._process.cpu_percent(interval=None)
        cpu_util = max(0.0, min(1.0, cpu_percent / 100.0)) if cpu_percent is not None else None
        gpu = get_gpu_snapshot(refresh=True)
        gpu_util = None
        if gpu.get("available") and gpu.get("utilization_percent") is not None:
            gpu_util = max(0.0, min(1.0, float(gpu["utilization_percent"]) / 100.0))
        return PressureSample(cpu_util=cpu_util, gpu_util=gpu_util)


class LLMGovernor:
    def __init__(
        self,
        *,
        enabled: bool,
        min_in_flight: int,
        max_in_flight: int,
        low_pressure_threshold: float,
        high_pressure_threshold: float,
        adjust_interval_s: float,
        provider: PressureProvider | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._min_limit = max(1, int(min_in_flight))
        self._max_limit = max(self._min_limit, int(max_in_flight))
        self._low_pressure = float(low_pressure_threshold)
        self._high_pressure = float(high_pressure_threshold)
        self._adjust_interval_s = max(0.1, float(adjust_interval_s))
        self._provider = provider or SystemPressureProvider()
        self._lock = threading.Condition()
        self._in_flight = 0
        self._foreground_waiters = 0
        self._background_waiters = 0
        self._current_limit = self._max_limit if self._enabled else 10_000
        self._last_adjust = 0.0

    @property
    def current_limit(self) -> int:
        return int(self._current_limit)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reserve(self, priority: str = "foreground") -> "_LLMReservation":
        return _LLMReservation(self, priority)

    def reserve_async(self, priority: str = "foreground") -> "_AsyncLLMReservation":
        return _AsyncLLMReservation(self, priority)

    def refresh(self) -> None:
        self._maybe_adjust_limit(force=True)

    def _normalize_priority(self, priority: str) -> str:
        value = (priority or "foreground").strip().lower()
        if value not in {"foreground", "background"}:
            return "foreground"
        return value

    def _maybe_adjust_limit(self, *, force: bool = False) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_adjust < self._adjust_interval_s:
            return
        sample = self._provider.sample()
        pressure = max(
            (val for val in (sample.cpu_util, sample.gpu_util) if val is not None),
            default=0.0,
        )
        if pressure >= self._high_pressure and self._current_limit > self._min_limit:
            self._current_limit -= 1
        elif pressure <= self._low_pressure and self._current_limit < self._max_limit:
            self._current_limit += 1
        self._last_adjust = now

    def _acquire(self, priority: str) -> None:
        if not self._enabled:
            return
        normalized = self._normalize_priority(priority)
        with self._lock:
            if normalized == "foreground":
                self._foreground_waiters += 1
            else:
                self._background_waiters += 1
            try:
                while True:
                    self._maybe_adjust_limit()
                    if self._in_flight < self._current_limit:
                        if normalized == "foreground" or self._foreground_waiters == 0:
                            self._in_flight += 1
                            return
                    self._lock.wait(timeout=0.5)
            finally:
                if normalized == "foreground":
                    self._foreground_waiters = max(0, self._foreground_waiters - 1)
                else:
                    self._background_waiters = max(0, self._background_waiters - 1)

    def _release(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._lock.notify_all()


class _LLMReservation:
    def __init__(self, governor: LLMGovernor, priority: str) -> None:
        self._governor = governor
        self._priority = priority

    def __enter__(self) -> "_LLMReservation":
        self._governor._acquire(self._priority)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._governor._release()


class _AsyncLLMReservation:
    def __init__(self, governor: LLMGovernor, priority: str) -> None:
        self._governor = governor
        self._priority = priority

    async def __aenter__(self) -> "_AsyncLLMReservation":
        await asyncio.to_thread(self._governor._acquire, self._priority)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._governor._release()


_governor_lock = threading.Lock()
_global_governor: LLMGovernor | None = None


def get_global_governor(config) -> LLMGovernor:
    global _global_governor
    with _governor_lock:
        if _global_governor is None:
            settings = getattr(config, "llm_governor", None)
            if settings is None:
                _global_governor = LLMGovernor(
                    enabled=True,
                    min_in_flight=1,
                    max_in_flight=4,
                    low_pressure_threshold=0.45,
                    high_pressure_threshold=0.85,
                    adjust_interval_s=5.0,
                )
            else:
                _global_governor = LLMGovernor(
                    enabled=settings.enabled,
                    min_in_flight=settings.min_in_flight,
                    max_in_flight=settings.max_in_flight,
                    low_pressure_threshold=settings.low_pressure_threshold,
                    high_pressure_threshold=settings.high_pressure_threshold,
                    adjust_interval_s=settings.adjust_interval_s,
                )
        return _global_governor
