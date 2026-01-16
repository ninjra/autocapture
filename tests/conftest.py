from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest
import asyncio
import fastapi.concurrency
import fastapi.routing
import starlette.concurrency

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("AUTOCAPTURE_TEST_MODE", "1")


@pytest.fixture
def async_client_factory():
    def _factory(app):
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://testserver")

    return _factory


@pytest.fixture(autouse=True)
def _disable_threadpool(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run_in_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(fastapi.concurrency, "run_in_threadpool", _run_in_threadpool)
    monkeypatch.setattr(fastapi.routing, "run_in_threadpool", _run_in_threadpool)
    monkeypatch.setattr(starlette.concurrency, "run_in_threadpool", _run_in_threadpool)

    async def _to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _to_thread)
