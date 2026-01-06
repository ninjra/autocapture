"""Resilience primitives for retries and circuit breaking."""

from __future__ import annotations

import asyncio
import random
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay_s: float = 0.25
    max_delay_s: float = 5.0
    jitter: float = 0.2


class CircuitBreaker:
    """Thread-safe circuit breaker with OPEN/HALF_OPEN/CLOSED states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self, failure_threshold: int = 5, reset_timeout_s: float = 30.0
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout_s = reset_timeout_s
        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def allow(self) -> bool:
        with self._lock:
            if self._state == self.CLOSED:
                return True
            if self._state == self.OPEN:
                if self._opened_at is None:
                    return False
                if time.monotonic() - self._opened_at >= self._reset_timeout_s:
                    self._state = self.HALF_OPEN
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = self.CLOSED
            self._opened_at = None

    def record_failure(self, exc: Exception | None = None) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()


def _compute_delay(policy: RetryPolicy, attempt: int) -> float:
    base = min(policy.max_delay_s, policy.base_delay_s * (2**attempt))
    jitter = base * policy.jitter
    return max(0.0, base + random.uniform(-jitter, jitter))


def retry_sync(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool],
) -> T:
    last_exc: Exception | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= policy.max_retries or not is_retryable(exc):
                raise
            time.sleep(_compute_delay(policy, attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("retry_sync failed without exception")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool],
) -> T:
    last_exc: Exception | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= policy.max_retries or not is_retryable(exc):
                raise
            await asyncio.sleep(_compute_delay(policy, attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("retry_async failed without exception")


def is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def is_retryable_exception(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            httpx.TransportError,
            TimeoutError,
            ConnectionError,
            OSError,
            ResponseHandlingException,
        ),
    ):
        return True
    if isinstance(exc, UnexpectedResponse):
        status = exc.status_code
        if status is not None and is_retryable_http_status(int(status)):
            return True
    return False
