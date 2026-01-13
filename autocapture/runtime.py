"""Unified runtime controller for Autocapture."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import datetime as dt

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .capture.orchestrator import CaptureOrchestrator
from .capture.raw_input import RawInputListener
from .capture.backends.monitor_utils import set_process_dpi_awareness
from .config import AppConfig
from .embeddings.service import EmbeddingService
from .indexing.vector_index import VectorIndex
from .logging_utils import get_logger
from .media.store import MediaStore
from .observability.metrics import MetricsServer
from .promptops import PromptOpsRunner
from .settings_store import update_settings
from .storage.database import DatabaseManager
from .storage.retention import RetentionManager
from .tracking import HostVectorTracker
from .worker.supervisor import WorkerSupervisor
from .qdrant.sidecar import QdrantSidecar


class RetentionScheduler:
    def __init__(self, retention: RetentionManager, interval_s: float = 1800.0) -> None:
        self._retention = retention
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._log = get_logger("retention.scheduler")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._retention.enforce()
                self._retention.enforce_screenshot_ttl()
            except Exception as exc:  # pragma: no cover - defensive
                self._log.warning("Retention enforcement failed: {}", exc)
            self._stop.wait(self._interval_s)


class PromptOpsScheduler:
    def __init__(self, runner: PromptOpsRunner, cron: str) -> None:
        self._runner = runner
        self._cron = cron
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        trigger = CronTrigger.from_crontab(self._cron)
        self._scheduler.add_job(
            self._run_job,
            trigger=trigger,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=60,
        )
        self._scheduler.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._scheduler.shutdown(wait=False)
        self._started = False

    def _run_job(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._runner.run_once()
        finally:
            self._lock.release()


class AppRuntime:
    """Start/stop the entire Autocapture pipeline in one process."""

    def __init__(self, config: AppConfig) -> None:
        set_process_dpi_awareness()
        self._config = config
        self._log = get_logger("runtime")
        self._lock = threading.Lock()
        self._running = False

        self._db = DatabaseManager(config.database)
        self._retrieval_embedder = EmbeddingService(config.embed)
        self._vector_index = VectorIndex(config, self._retrieval_embedder.dim)
        self._tracker = (
            HostVectorTracker(
                config=config.tracking,
                data_dir=str(config.capture.data_dir),
                idle_grace_ms=config.capture.hid.idle_grace_ms,
            )
            if config.tracking.enabled
            else None
        )
        self._raw_input = RawInputListener(
            idle_grace_ms=config.capture.hid.idle_grace_ms,
            on_activity=None,
            on_hotkey=None,
            on_input_event=self._tracker.ingest_input_event if self._tracker else None,
            track_mouse_movement=config.tracking.track_mouse_movement,
            mouse_move_sample_ms=config.tracking.mouse_move_sample_ms,
        )
        self._orchestrator = CaptureOrchestrator(
            database=self._db,
            capture_config=config.capture,
            worker_config=config.worker,
            privacy_config=config.privacy,
            on_ocr_observation=self._on_ocr_observation,
            on_vision_observation=None,
            vision_sample_rate=getattr(config.capture, "vision_sample_rate", 0.0),
            raw_input=self._raw_input,
            ocr_backlog_check_s=1.0,
            media_store=MediaStore(config.capture, config.encryption),
            ffmpeg_config=config.ffmpeg,
        )
        self._qdrant_sidecar = QdrantSidecar(
            config, Path(config.capture.data_dir), Path(config.capture.data_dir) / "logs"
        )
        self._workers = WorkerSupervisor(
            config=config,
            db_manager=self._db,
            vector_index=self._vector_index,
        )
        self._retention = RetentionManager(
            config.storage,
            config.retention,
            self._db,
            Path(config.capture.data_dir),
        )
        self._retention_scheduler = RetentionScheduler(self._retention)
        self._metrics = MetricsServer(config.observability, Path(config.capture.data_dir))
        self._settings_path = Path(config.capture.data_dir) / "settings.json"
        self._snooze_timer: threading.Timer | None = None
        self._promptops_runner: PromptOpsRunner | None = None
        self._promptops_scheduler: PromptOpsScheduler | None = None
        if config.promptops.enabled:
            self._promptops_runner = PromptOpsRunner(config, self._db)
            self._promptops_scheduler = PromptOpsScheduler(
                self._promptops_runner, config.promptops.schedule_cron
            )

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        self._log.info("Runtime starting")
        self._qdrant_sidecar.start()
        if self._tracker:
            self._tracker.start()
        self._orchestrator.start()
        self._workers.start()
        self._retention_scheduler.start()
        self._metrics.start()
        if self._promptops_scheduler:
            self._promptops_scheduler.start()
        self._apply_startup_pause()
        self._log.info("Runtime started")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False

        self._log.info("Runtime stopping")
        try:
            try:
                self._orchestrator.stop()
            except Exception as exc:  # pragma: no cover - defensive
                self._log.warning("Failed to stop orchestrator: {}", exc)

            try:
                self._workers.stop()
            finally:
                self._workers.flush()

            self._metrics.stop()
            if self._tracker:
                self._tracker.stop()
            self._retention_scheduler.stop()
            if self._promptops_scheduler:
                self._promptops_scheduler.stop()
            self._cancel_snooze_timer()
        finally:
            self._qdrant_sidecar.stop()
        self._log.info("Runtime stopped")

    def pause_capture(self) -> None:
        self._cancel_snooze_timer()
        self._config.privacy.paused = True
        self._config.privacy.snooze_until_utc = None
        self._persist_privacy()
        self._orchestrator.pause()

    def resume_capture(self) -> None:
        self._cancel_snooze_timer()
        self._config.privacy.paused = False
        self._config.privacy.snooze_until_utc = None
        self._persist_privacy()
        self._orchestrator.resume()

    def set_hotkey_callback(self, callback) -> None:
        self._raw_input.set_hotkey_callback(callback)

    def _on_ocr_observation(self, observation_id: str) -> None:
        self._workers.notify_ocr_observation(observation_id)

    def wait_forever(self) -> None:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self._log.info("Runtime interrupted")

    def _apply_startup_pause(self) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        snooze_until = self._ensure_aware(self._config.privacy.snooze_until_utc)
        if snooze_until and snooze_until > now:
            self._config.privacy.snooze_until_utc = snooze_until
        should_pause = bool(self._config.privacy.paused)
        if snooze_until and snooze_until > now:
            should_pause = True
            self._schedule_snooze_resume(snooze_until)
        if should_pause:
            self._orchestrator.pause()

    def _schedule_snooze_resume(self, until: dt.datetime) -> None:
        delay = max((until - dt.datetime.now(dt.timezone.utc)).total_seconds(), 0.0)
        self._cancel_snooze_timer()
        timer = threading.Timer(delay, self._auto_resume)
        timer.daemon = True
        self._snooze_timer = timer
        timer.start()

    def _auto_resume(self) -> None:
        if not self._running:
            return
        self._log.info("Snooze expired; resuming capture")
        self._config.privacy.paused = False
        self._config.privacy.snooze_until_utc = None
        self._persist_privacy()
        self._orchestrator.resume()

    def _cancel_snooze_timer(self) -> None:
        if self._snooze_timer:
            self._snooze_timer.cancel()
            self._snooze_timer = None

    def _persist_privacy(self) -> None:
        def _update(settings: dict) -> dict:
            privacy = settings.get("privacy")
            if not isinstance(privacy, dict):
                privacy = {}
            privacy["paused"] = self._config.privacy.paused
            privacy["snooze_until_utc"] = _to_iso(self._config.privacy.snooze_until_utc)
            settings["privacy"] = privacy
            return settings

        update_settings(self._settings_path, _update)

    @staticmethod
    def _ensure_aware(value: dt.datetime | None) -> dt.datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value


def _to_iso(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.isoformat()
