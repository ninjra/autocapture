"""App lifespan wiring."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import time

from fastapi import FastAPI

from ..logging_utils import get_logger
from ..ux.state_store import HeartbeatWriter
from .container import AppContainer


def build_lifespan(container: AppContainer):
    log = get_logger("api")
    retention = container.retention
    db = container.db
    db_owned = container.db_owned
    state_dir = Path(container.config.capture.data_dir) / "state"
    heartbeat_path = state_dir / "api.json"
    start_time = time.monotonic()
    heartbeat = HeartbeatWriter(
        heartbeat_path,
        component="api",
        interval_s=2.0,
        build_payload=lambda: (
            "ok",
            {
                "mode": container.config.mode.mode,
                "uptime_s": max(time.monotonic() - start_time, 0.0),
            },
            [],
        ),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await asyncio.to_thread(retention.enforce_screenshot_ttl)
        except Exception as exc:
            log.warning("Retention enforcement failed: {}", exc)
        heartbeat.start()
        yield
        heartbeat.stop()
        if db_owned:
            try:
                db.engine.dispose()
            except Exception as exc:
                log.warning("Database dispose failed: {}", exc)

    return lifespan
