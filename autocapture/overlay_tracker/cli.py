"""CLI helpers for overlay tracker."""

from __future__ import annotations

from ..config import AppConfig
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from .service import OverlayTrackerService
from .store import OverlayTrackerStore
from .clock import SystemClock


def overlay_status(config: AppConfig) -> int:
    log = get_logger("overlay_tracker.cli")
    db = DatabaseManager(config.database)
    service = OverlayTrackerService(
        config.overlay_tracker,
        db,
        sanitize=config.privacy.sanitize_default,
    )
    store = OverlayTrackerStore(db, SystemClock())
    health = service.health()
    retention = store.get_kv("retention_last_run")
    log.info("Overlay tracker status: {}", health)
    if retention:
        log.info("Overlay tracker retention: {}", retention)
    return 0
