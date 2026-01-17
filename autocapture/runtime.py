"""Unified runtime controller for Autocapture."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import datetime as dt

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .agents import AGENT_JOB_DAILY_HIGHLIGHTS, AGENT_JOB_VISION_CAPTION
from .agents.jobs import AgentJobQueue
from .capture.orchestrator import CaptureOrchestrator
from .capture.privacy_filter import normalize_process_name
from .capture.raw_input import RawInputListener
from .capture.backends.monitor_utils import set_process_dpi_awareness
from .config import AppConfig
from .embeddings.service import EmbeddingService
from .enrichment.scheduler import EnrichmentScheduler
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
from .runtime_governor import RuntimeGovernor, RuntimeMode
from .gpu_lease import get_global_gpu_lease


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


class HighlightsScheduler:
    def __init__(self, queue: AgentJobQueue, cron: str, max_pending: int) -> None:
        self._queue = queue
        self._cron = cron
        self._max_pending = max_pending
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        trigger = CronTrigger.from_crontab(self._cron)
        self._scheduler.add_job(
            self._enqueue_job,
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

    def _enqueue_job(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            day = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).date().isoformat()
            job_key = f"highlights:{day}:v1"
            self._queue.enqueue(
                job_key=job_key,
                job_type=AGENT_JOB_DAILY_HIGHLIGHTS,
                day=day,
                payload={"day": day},
                max_pending=self._max_pending,
            )
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
        self._gpu_lease = get_global_gpu_lease()
        self._agent_jobs = AgentJobQueue(self._db)
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
        self._runtime_governor = RuntimeGovernor(
            config.runtime,
            db_manager=self._db,
            raw_input=self._raw_input,
            gpu_lease=self._gpu_lease,
        )
        self._orchestrator = CaptureOrchestrator(
            database=self._db,
            capture_config=config.capture,
            worker_config=config.worker,
            privacy_config=config.privacy,
            on_ocr_observation=self._on_ocr_observation,
            on_vision_observation=self._on_vision_observation,
            vision_sample_rate=getattr(config.capture, "vision_sample_rate", 0.0),
            raw_input=self._raw_input,
            ocr_backlog_check_s=1.0,
            media_store=MediaStore(config.capture, config.encryption),
            ffmpeg_config=config.ffmpeg,
            runtime_governor=self._runtime_governor,
            runtime_auto_pause=config.runtime.auto_pause.on_fullscreen,
        )
        self._qdrant_sidecar = QdrantSidecar(
            config, Path(config.capture.data_dir), Path(config.capture.data_dir) / "logs"
        )
        self._workers = WorkerSupervisor(
            config=config,
            db_manager=self._db,
            vector_index=self._vector_index,
            runtime_governor=self._runtime_governor,
        )
        self._retention = RetentionManager(
            config.storage,
            config.retention,
            self._db,
            Path(config.capture.data_dir),
        )
        self._retention_scheduler = RetentionScheduler(self._retention)
        self._enrichment_scheduler = EnrichmentScheduler(
            config,
            self._db,
            self._agent_jobs,
            embedder=self._retrieval_embedder,
            vector_index=self._vector_index,
        )
        self._metrics = MetricsServer(config.observability, Path(config.capture.data_dir))
        self._settings_path = Path(config.capture.data_dir) / "settings.json"
        self._snooze_timer: threading.Timer | None = None
        self._promptops_runner: PromptOpsRunner | None = None
        self._promptops_scheduler: PromptOpsScheduler | None = None
        self._highlights_scheduler: HighlightsScheduler | None = None
        self._fullscreen_paused = False
        if config.promptops.enabled:
            self._promptops_runner = PromptOpsRunner(config, self._db)
            self._promptops_scheduler = PromptOpsScheduler(
                self._promptops_runner, config.promptops.schedule_cron
            )
        if config.agents.enabled:
            self._highlights_scheduler = HighlightsScheduler(
                self._agent_jobs, config.agents.nightly_cron, config.agents.max_pending_jobs
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
        self._runtime_governor.subscribe(self._on_runtime_mode_change)
        self._runtime_governor.start()
        self._workers.start()
        self._retention_scheduler.start()
        if self._enrichment_scheduler:
            self._enrichment_scheduler.start()
        self._metrics.start()
        if self._promptops_scheduler:
            self._promptops_scheduler.start()
        if self._highlights_scheduler:
            self._highlights_scheduler.start()
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

            self._runtime_governor.unsubscribe(self._on_runtime_mode_change)
            self._runtime_governor.stop()
            self._metrics.stop()
            if self._tracker:
                self._tracker.stop()
            self._retention_scheduler.stop()
            if self._enrichment_scheduler:
                self._enrichment_scheduler.stop()
            if self._promptops_scheduler:
                self._promptops_scheduler.stop()
            if self._highlights_scheduler:
                self._highlights_scheduler.stop()
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

    def snooze_capture(self, minutes: int) -> None:
        if minutes <= 0:
            return
        self._cancel_snooze_timer()
        until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=minutes)
        self._config.privacy.paused = True
        self._config.privacy.snooze_until_utc = until
        self._persist_privacy()
        self._orchestrator.pause()
        self._schedule_snooze_resume(until)

    def add_excluded_process(self, process_name: str) -> bool:
        normalized = normalize_process_name(process_name)
        if not normalized:
            return False
        existing = {normalize_process_name(name) for name in self._config.privacy.exclude_processes}
        if normalized in existing:
            return False
        self._config.privacy.exclude_processes.append(normalized)
        self._persist_privacy()
        return True

    def add_excluded_window_title_regex(self, pattern: str) -> bool:
        if not pattern:
            return False
        if pattern in self._config.privacy.exclude_window_title_regex:
            return False
        self._config.privacy.exclude_window_title_regex.append(pattern)
        self._persist_privacy()
        return True

    def set_hotkey_callback(self, callback) -> None:
        self._raw_input.set_hotkey_callback(callback)

    def _on_ocr_observation(self, observation_id: str) -> None:
        self._workers.notify_ocr_observation(observation_id)

    def _on_vision_observation(self, observation_id: str) -> None:
        if not self._config.agents.enabled:
            return
        job_key = f"vision:{observation_id}:v1"
        self._agent_jobs.enqueue(
            job_key=job_key,
            job_type=AGENT_JOB_VISION_CAPTION,
            event_id=observation_id,
            payload={"event_id": observation_id},
            max_attempts=2,
            max_pending=self._config.agents.max_pending_jobs,
        )

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

    def _on_runtime_mode_change(self, mode: RuntimeMode) -> None:
        if mode == RuntimeMode.FULLSCREEN_HARD_PAUSE:
            if not self._fullscreen_paused:
                self._fullscreen_paused = True
                self._orchestrator.pause()
                self._pause_background_loops()
            return
        if self._fullscreen_paused:
            self._fullscreen_paused = False
            if not self._config.privacy.paused:
                self._orchestrator.resume()
            self._resume_background_loops()

    def _pause_background_loops(self) -> None:
        try:
            self._retention_scheduler.stop()
        except Exception:
            pass
        if self._enrichment_scheduler:
            try:
                self._enrichment_scheduler.stop()
            except Exception:
                pass
        if self._promptops_scheduler:
            try:
                self._promptops_scheduler.stop()
            except Exception:
                pass
        if self._highlights_scheduler:
            try:
                self._highlights_scheduler.stop()
            except Exception:
                pass

    def _resume_background_loops(self) -> None:
        try:
            self._retention_scheduler.start()
        except Exception:
            pass
        if self._enrichment_scheduler:
            try:
                self._enrichment_scheduler.start()
            except Exception:
                pass
        if self._promptops_scheduler:
            try:
                self._promptops_scheduler.start()
            except Exception:
                pass
        if self._highlights_scheduler:
            try:
                self._highlights_scheduler.start()
            except Exception:
                pass

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
            privacy["exclude_monitors"] = list(self._config.privacy.exclude_monitors)
            privacy["exclude_processes"] = list(self._config.privacy.exclude_processes)
            privacy["exclude_window_title_regex"] = list(
                self._config.privacy.exclude_window_title_regex
            )
            privacy["exclude_regions"] = list(self._config.privacy.exclude_regions)
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
