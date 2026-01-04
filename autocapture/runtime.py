"""Unified runtime controller for Autocapture."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn

from .api.server import create_app
from .capture.orchestrator import CaptureOrchestrator
from .capture.raw_input import RawInputListener
from .config import AppConfig
from .logging_utils import get_logger
from .media.store import MediaStore
from .observability.metrics import MetricsServer
from .storage.database import DatabaseManager
from .storage.retention import RetentionManager
from .tracking import HostVectorTracker
from .worker.supervisor import WorkerSupervisor


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
                self._log.warning("Retention enforcement failed: %s", exc)
            self._stop.wait(self._interval_s)


class AppRuntime:
    """Start/stop the entire Autocapture pipeline in one process."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._log = get_logger("runtime")
        self._lock = threading.Lock()
        self._running = False

        self._db = DatabaseManager(config.database)
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
            on_hotkey=self._toggle_capture,
            on_input_event=self._tracker.ingest_input_event if self._tracker else None,
            track_mouse_movement=config.tracking.track_mouse_movement,
            mouse_move_sample_ms=config.tracking.mouse_move_sample_ms,
        )
        self._orchestrator = CaptureOrchestrator(
            database=self._db,
            data_dir=Path(config.capture.data_dir),
            idle_grace_ms=config.capture.hid.idle_grace_ms,
            fps_soft_cap=config.capture.hid.fps_soft_cap,
            on_ocr_observation=self._on_ocr_observation,
            on_vision_observation=None,
            vision_sample_rate=getattr(config.capture, "vision_sample_rate", 0.0),
            raw_input=self._raw_input,
            ocr_backlog_soft_limit=config.worker.ocr_backlog_soft_limit,
            ocr_backlog_check_s=1.0,
            media_store=MediaStore(config.capture, config.encryption),
        )
        self._workers = WorkerSupervisor(config=config, db_manager=self._db)
        self._retention = RetentionManager(
            config.storage,
            config.retention,
            self._db,
            Path(config.capture.data_dir),
        )
        self._retention_scheduler = RetentionScheduler(self._retention)
        self._metrics = MetricsServer(config.observability, Path(config.capture.data_dir))
        self._api_server: Optional[uvicorn.Server] = None
        self._api_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        self._log.info("Runtime starting")
        if self._tracker:
            self._tracker.start()
        self._orchestrator.start()
        self._workers.start()
        self._retention_scheduler.start()
        self._metrics.start()
        self._start_api()
        self._log.info("Runtime started")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False

        self._log.info("Runtime stopping")
        try:
            self._orchestrator.stop()
        except Exception as exc:  # pragma: no cover - defensive
            self._log.warning("Failed to stop orchestrator: %s", exc)

        try:
            self._workers.stop()
        finally:
            self._workers.flush()

        self._stop_api()
        self._metrics.stop()
        if self._tracker:
            self._tracker.stop()
        self._retention_scheduler.stop()
        self._log.info("Runtime stopped")

    def pause_capture(self) -> None:
        self._orchestrator.pause()

    def resume_capture(self) -> None:
        self._orchestrator.resume()

    def _toggle_capture(self) -> None:
        if self._orchestrator.is_paused:
            self.resume_capture()
        else:
            self.pause_capture()

    def _on_ocr_observation(self, observation_id: str) -> None:
        self._workers.notify_ocr_observation(observation_id)

    def _start_api(self) -> None:
        app = create_app(self._config, db_manager=self._db)
        config = uvicorn.Config(
            app,
            host=self._config.api.bind_host,
            port=self._config.api.port,
            log_level="info",
            ssl_certfile=(
                str(self._config.mode.tls_cert_path)
                if self._config.mode.https_enabled
                else None
            ),
            ssl_keyfile=(
                str(self._config.mode.tls_key_path)
                if self._config.mode.https_enabled
                else None
            ),
        )
        self._api_server = uvicorn.Server(config)
        self._api_thread = threading.Thread(target=self._api_server.run, daemon=True)
        self._api_thread.start()

    def _stop_api(self) -> None:
        if not self._api_server:
            return
        self._api_server.should_exit = True
        if self._api_thread:
            self._api_thread.join(timeout=3.0)
        self._api_server = None
        self._api_thread = None

    def wait_forever(self) -> None:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self._log.info("Runtime interrupted")
