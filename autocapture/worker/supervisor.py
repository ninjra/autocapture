"""Worker supervisor for OCR ingest and embedding jobs."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..config import AppConfig, RuntimeQosProfile, is_dev_mode
from ..logging_utils import get_logger
from ..observability.metrics import worker_restarts_total
from ..storage.database import DatabaseManager
from ..indexing.vector_index import VectorIndex
from ..runtime_governor import RuntimeGovernor, RuntimeMode
from ..runtime_qos import apply_cpu_priority
from .agent_worker import AgentJobWorker
from .embedding_worker import EmbeddingWorker
from .event_worker import EventIngestWorker


@dataclass
class _WorkerSlot:
    worker: object
    thread: threading.Thread | None
    stop_event: threading.Event
    name: str
    worker_type: str


class WorkerSupervisor:
    def __init__(
        self,
        config: AppConfig,
        db_manager: DatabaseManager | None = None,
        vector_index: VectorIndex | None = None,
        *,
        ocr_workers: list[object] | None = None,
        embed_workers: list[object] | None = None,
        agent_workers: list[object] | None = None,
        runtime_governor: RuntimeGovernor | None = None,
    ) -> None:
        self._config = config
        self._db = db_manager or DatabaseManager(config.database)
        self._log = get_logger("worker.supervisor")
        self._shutdown_event = threading.Event()
        self._ocr_slots: list[_WorkerSlot] = []
        self._embed_slots: list[_WorkerSlot] = []
        self._agent_slots: list[_WorkerSlot] = []
        self._watchdog_thread: threading.Thread | None = None
        self._slots_lock = threading.Lock()
        self._desired_counts = {"ocr": 0, "embed": 0, "agents": 0}
        self._runtime = runtime_governor
        if ocr_workers is None:
            if self._ocr_enabled():
                self._ocr_workers = [
                    EventIngestWorker(
                        config, db_manager=self._db, runtime_governor=runtime_governor
                    )
                    for _ in range(config.worker.ocr_workers)
                ]
            else:
                self._ocr_workers = []
                self._log.info("OCR workers disabled.")
        else:
            self._ocr_workers = list(ocr_workers)
        if embed_workers is None:
            self._embed_workers = [
                EmbeddingWorker(
                    config,
                    db_manager=self._db,
                    vector_index=vector_index,
                    runtime_governor=runtime_governor,
                )
                for _ in range(config.worker.embed_workers)
            ]
        else:
            self._embed_workers = list(embed_workers)
        if agent_workers is None:
            self._agent_workers = [
                AgentJobWorker(
                    config,
                    db_manager=self._db,
                    vector_index=vector_index,
                    runtime_governor=runtime_governor,
                )
                for _ in range(config.worker.agent_workers)
            ]
        else:
            self._agent_workers = list(agent_workers)

        self._ocr_slots = [
            _WorkerSlot(
                worker=worker,
                thread=None,
                stop_event=threading.Event(),
                name=f"ocr-worker-{idx}",
                worker_type="ocr",
            )
            for idx, worker in enumerate(self._ocr_workers)
        ]
        self._embed_slots = [
            _WorkerSlot(
                worker=worker,
                thread=None,
                stop_event=threading.Event(),
                name=f"embed-worker-{idx}",
                worker_type="embed",
            )
            for idx, worker in enumerate(self._embed_workers)
        ]
        self._agent_slots = [
            _WorkerSlot(
                worker=worker,
                thread=None,
                stop_event=threading.Event(),
                name=f"agent-worker-{idx}",
                worker_type="agents",
            )
            for idx, worker in enumerate(self._agent_workers)
        ]

    def _ocr_enabled(self) -> bool:
        engine = (self._config.vision_extract.engine or "").lower()
        if self._config.routing.ocr == "disabled" or engine in {"disabled", "off"}:
            return False
        if engine not in {"rapidocr", "rapidocr-onnxruntime"}:
            return True
        if self._config.ocr.engine == "disabled":
            return False
        if not is_dev_mode():
            return True
        import importlib.util

        if importlib.util.find_spec("rapidocr_onnxruntime") is None:
            self._log.warning("Dev mode: rapidocr_onnxruntime missing; skipping OCR workers.")
            return False
        return True

    def start(self) -> None:
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        with self._slots_lock:
            if any(
                slot.thread and slot.thread.is_alive()
                for slot in self._ocr_slots + self._embed_slots + self._agent_slots
            ):
                return
        self._shutdown_event.clear()
        self._apply_profile(self._initial_profile())
        if self._runtime:
            self._runtime.subscribe(self._on_mode_change)
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="worker-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()
        self._log.info("Worker supervisor started")

    def stop(self) -> None:
        if self._runtime:
            self._runtime.unsubscribe(self._on_mode_change)
        self._shutdown_event.set()
        with self._slots_lock:
            slots = list(self._ocr_slots + self._embed_slots + self._agent_slots)
        for slot in slots:
            slot.stop_event.set()
            if slot.thread:
                slot.thread.join(timeout=2.0)
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
        self._log.info("Worker supervisor stopped")

    def flush(self) -> None:
        """Flush any pending worker queues (best-effort)."""

    def notify_ocr_observation(self, observation_id: str) -> None:
        # Placeholder for future queue-based workers.
        _ = observation_id

    def health_snapshot(self) -> dict[str, bool]:
        watchdog_alive = bool(self._watchdog_thread and self._watchdog_thread.is_alive())
        with self._slots_lock:
            worker_threads = self._ocr_slots + self._embed_slots + self._agent_slots
        workers_alive = all(slot.thread and slot.thread.is_alive() for slot in worker_threads)
        return {
            "watchdog_alive": watchdog_alive,
            "workers_alive": workers_alive,
        }

    def _build_thread(
        self, worker: object, name: str, stop_event: threading.Event
    ) -> threading.Thread:
        return threading.Thread(
            target=getattr(worker, "run_forever"),
            kwargs={"stop_event": stop_event},
            daemon=True,
            name=name,
        )

    def _watchdog_loop(self) -> None:
        interval = self._config.worker.watchdog_interval_s
        while not self._shutdown_event.wait(interval):
            with self._slots_lock:
                slots = list(self._ocr_slots + self._embed_slots + self._agent_slots)
                desired = dict(self._desired_counts)
            for idx, slot in enumerate(slots):
                if not self._slot_should_run(slot, idx, desired):
                    continue
                if slot.thread and slot.thread.is_alive():
                    continue
                if slot.stop_event.is_set() or self._shutdown_event.is_set():
                    continue
                self._log.error("Worker thread {} died; restarting", slot.name)
                worker_restarts_total.labels(slot.worker_type).inc()
                slot.thread = self._build_thread(slot.worker, slot.name, slot.stop_event)
                slot.thread.start()

    def _slot_should_run(self, slot: _WorkerSlot, idx: int, desired: dict[str, int]) -> bool:
        if slot.worker_type == "ocr":
            return idx < desired["ocr"]
        if slot.worker_type == "embed":
            return idx < desired["embed"]
        return idx < desired["agents"]

    def _initial_profile(self) -> RuntimeQosProfile | None:
        if not self._runtime:
            return RuntimeQosProfile(
                ocr_workers=self._config.worker.ocr_workers,
                embed_workers=self._config.worker.embed_workers,
                agent_workers=self._config.worker.agent_workers,
                vision_extract=True,
                ui_grounding=True,
                cpu_priority="normal",
            )
        return self._runtime.qos_profile()

    def _on_mode_change(self, mode: RuntimeMode) -> None:
        if mode == RuntimeMode.FULLSCREEN_HARD_PAUSE:
            self._apply_profile(None)
            return
        profile = self._runtime.qos_profile(mode) if self._runtime else None
        self._apply_profile(profile)

    def _apply_profile(self, profile: RuntimeQosProfile | None) -> None:
        if profile is None:
            desired = {"ocr": 0, "embed": 0, "agents": 0}
        else:
            desired = {
                "ocr": max(0, int(profile.ocr_workers)),
                "embed": max(0, int(profile.embed_workers)),
                "agents": max(0, int(profile.agent_workers)),
            }
            apply_cpu_priority(profile.cpu_priority)
        with self._slots_lock:
            self._desired_counts = desired
            self._apply_worker_counts(self._ocr_slots, desired["ocr"])
            self._apply_worker_counts(self._embed_slots, desired["embed"])
            self._apply_worker_counts(self._agent_slots, desired["agents"])

    def _apply_worker_counts(self, slots: list[_WorkerSlot], desired: int) -> None:
        for idx, slot in enumerate(slots):
            if idx < desired:
                if slot.thread and slot.thread.is_alive():
                    continue
                slot.stop_event = threading.Event()
                slot.thread = self._build_thread(slot.worker, slot.name, slot.stop_event)
                slot.thread.start()
            else:
                if slot.thread and slot.thread.is_alive():
                    slot.stop_event.set()
                    slot.thread.join(timeout=2.0)
                slot.thread = None
