"""App lifespan wiring."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..logging_utils import get_logger
from .container import AppContainer


def build_lifespan(container: AppContainer):
    log = get_logger("api")
    retention = container.retention
    db = container.db
    db_owned = container.db_owned

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await asyncio.to_thread(retention.enforce_screenshot_ttl)
        except Exception as exc:
            log.warning("Retention enforcement failed: {}", exc)
        yield
        if db_owned:
            try:
                db.engine.dispose()
            except Exception as exc:
                log.warning("Database dispose failed: {}", exc)

    return lifespan
