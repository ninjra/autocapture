"""Worker supervisor for OCR ingest and embedding jobs."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..config import AppConfig
from ..logging_utils import get_logger
from ..observability.metrics import worker_restarts_total
from ..storage.database import DatabaseManager
from ..indexing.vector_index import VectorIndex
from .agent_worker import AgentJobWorker
from .embedding_worker import EmbeddingWorker
from .event_worker import EventIngestWorker


@dataclass
class _WorkerSlot:
    worker: object
    thread: threading.Thread
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
    ) -> None:
        self._config = config
        self._db = db_manager or DatabaseManager(config.database)
        self._log = get_logger("worker.supervisor")
        self._stop_event = threading.Event()
        self._ocr_threads: list[_WorkerSlot] = []
        self._embed_threads: list[_WorkerSlot] = []
        self._agent_threads: list[_WorkerSlot] = []
        self._watchdog_thread: threading.Thread | None = None
        if ocr_workers is None:
            self._ocr_workers = [
                EventIngestWorker(config, db_manager=self._db)
                for _ in range(config.worker.ocr_workers)
            ]
        else:
            self._ocr_workers = list(ocr_workers)
        if embed_workers is None:
            self._embed_workers = [
                EmbeddingWorker(
                    config,
                    db_manager=self._db,
                    vector_index=vector_index,
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
                )
                for _ in range(config.worker.agent_workers)
            ]
        else:
            self._agent_workers = list(agent_workers)

    def start(self) -> None:
        if any(
            slot.thread.is_alive()
            for slot in self._ocr_threads + self._embed_threads + self._agent_threads
        ):
            return
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._stop_event.clear()
        self._ocr_threads = [
            _WorkerSlot(
                worker=worker,
                thread=self._build_thread(worker, f"ocr-worker-{idx}"),
                name=f"ocr-worker-{idx}",
                worker_type="ocr",
            )
            for idx, worker in enumerate(self._ocr_workers)
        ]
        self._embed_threads = [
            _WorkerSlot(
                worker=worker,
                thread=self._build_thread(worker, f"embed-worker-{idx}"),
                name=f"embed-worker-{idx}",
                worker_type="embed",
            )
            for idx, worker in enumerate(self._embed_workers)
        ]
        self._agent_threads = [
            _WorkerSlot(
                worker=worker,
                thread=self._build_thread(worker, f"agent-worker-{idx}"),
                name=f"agent-worker-{idx}",
                worker_type="agents",
            )
            for idx, worker in enumerate(self._agent_workers)
        ]
        for slot in self._ocr_threads + self._embed_threads + self._agent_threads:
            slot.thread.start()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="worker-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()
        self._log.info("Worker supervisor started")

    def stop(self) -> None:
        self._stop_event.set()
        for slot in self._ocr_threads + self._embed_threads + self._agent_threads:
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
        worker_threads = self._ocr_threads + self._embed_threads + self._agent_threads
        workers_alive = all(slot.thread.is_alive() for slot in worker_threads)
        return {
            "watchdog_alive": watchdog_alive,
            "workers_alive": workers_alive,
        }

    def _build_thread(self, worker: object, name: str) -> threading.Thread:
        return threading.Thread(
            target=getattr(worker, "run_forever"),
            kwargs={"stop_event": self._stop_event},
            daemon=True,
            name=name,
        )

    def _watchdog_loop(self) -> None:
        interval = self._config.worker.watchdog_interval_s
        while not self._stop_event.wait(interval):
            for slot in self._ocr_threads + self._embed_threads + self._agent_threads:
                if slot.thread.is_alive():
                    continue
                self._log.error("Worker thread {} died; restarting", slot.name)
                worker_restarts_total.labels(slot.worker_type).inc()
                slot.thread = self._build_thread(slot.worker, slot.name)
                slot.thread.start()
