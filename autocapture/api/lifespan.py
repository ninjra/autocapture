"""App lifespan wiring."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import time

from fastapi import FastAPI

from ..logging_utils import get_logger
from ..observability.perf_logger import PerfLogger
from ..observability.perf_snapshot import PerfSnapshotBuilder
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
    perf_builder = PerfSnapshotBuilder(
        component="api",
        include_metrics=False,
        include_gpu=False,
    )
    perf_logger = PerfLogger(
        Path(container.config.capture.data_dir),
        "api",
        perf_builder.snapshot,
    )
    heartbeat = HeartbeatWriter(
        heartbeat_path,
        component="api",
        interval_s=2.0,
        build_payload=lambda: (
            "ok",
            {
                "mode": container.config.mode.mode,
                "uptime_s": max(time.monotonic() - start_time, 0.0),
                "perf": perf_builder.snapshot(
                    extra={"uptime_s": max(time.monotonic() - start_time, 0.0)}
                ),
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
        perf_logger.start()
        heartbeat.start()
        yield
        heartbeat.stop()
        perf_logger.stop()
        if db_owned:
            try:
                db.engine.dispose()
            except Exception as exc:
                log.warning("Database dispose failed: {}", exc)

    return lifespan
