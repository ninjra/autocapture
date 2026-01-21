"""Rate limit middleware."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

from fastapi import Request
from fastapi.responses import JSONResponse


class _TokenBucket:
    def __init__(self, tokens: float, updated_at: float) -> None:
        self.tokens = tokens
        self.updated_at = updated_at


class RateLimiter:
    def __init__(self, rps: float, burst: int, max_entries: int = 10_000) -> None:
        self._rps = rps
        self._burst = burst
        self._max_entries = max_entries
        self._buckets: OrderedDict[str, _TokenBucket] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _TokenBucket(tokens=self._burst, updated_at=now)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(self._burst, bucket.tokens + elapsed * self._rps)
                bucket.updated_at = now
                self._buckets.move_to_end(key)
            if len(self._buckets) > self._max_entries:
                self._buckets.popitem(last=False)
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            retry_after = max(0.0, (1.0 - bucket.tokens) / self._rps)
            return False, retry_after


def install_rate_limit_middleware(
    app, *, rate_limiter: RateLimiter, rate_limited_paths: set[str]
) -> None:
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):  # type: ignore[no-redef]
        if request.url.path in rate_limited_paths and request.method == "POST":
            client_host = request.client.host if request.client else "unknown"
            key = f"{client_host}:{request.url.path}"
            allowed, retry_after = rate_limiter.allow(key)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": f"{int(retry_after) + 1}"},
                    content={"detail": "Rate limit exceeded"},
                )
        return await call_next(request)
