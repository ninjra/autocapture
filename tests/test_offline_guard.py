from __future__ import annotations

import pytest
import httpx

from autocapture.security.offline_guard import apply_offline_guard


def test_offline_guard_blocks_remote() -> None:
    apply_offline_guard({"127.0.0.1", "::1", "localhost"}, enabled=True)
    with pytest.raises(RuntimeError):
        httpx.get("https://example.com", timeout=1.0)


def test_offline_guard_allows_loopback() -> None:
    apply_offline_guard({"127.0.0.1", "::1", "localhost"}, enabled=True)
    with pytest.raises(httpx.TransportError):
        httpx.get("http://127.0.0.1:1234", timeout=0.5)
