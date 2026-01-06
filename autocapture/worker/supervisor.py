"""Worker supervisor for OCR ingest and embedding jobs."""

from __future__ import annotations

import threading

from ..config import AppConfig
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..indexing.vector_index import VectorIndex
from .embedding_worker import EmbeddingWorker
from .event_worker import EventIngestWorker


class WorkerSupervisor:
    def __init__(
        self,
        config: AppConfig,
        db_manager: DatabaseManager | None = None,
        vector_index: VectorIndex | None = None,
    ) -> None:
        self._config = config
        self._db = db_manager or DatabaseManager(config.database)
        self._log = get_logger("worker.supervisor")
        self._stop_event = threading.Event()
        self._ocr_threads: list[threading.Thread] = []
        self._embed_threads: list[threading.Thread] = []
        self._ocr_workers = [
            EventIngestWorker(config, db_manager=self._db)
            for _ in range(config.worker.ocr_workers)
        ]
        self._embed_workers = [
            EmbeddingWorker(
                config,
                db_manager=self._db,
                vector_index=vector_index,
            )
            for _ in range(config.worker.embed_workers)
        ]

    def start(self) -> None:
        if any(thread.is_alive() for thread in self._ocr_threads + self._embed_threads):
            return
        self._stop_event.clear()
        self._ocr_threads = [
            threading.Thread(
                target=worker.run_forever,
                kwargs={"stop_event": self._stop_event},
                daemon=True,
            )
            for worker in self._ocr_workers
        ]
        self._embed_threads = [
            threading.Thread(
                target=worker.run_forever,
                kwargs={"stop_event": self._stop_event},
                daemon=True,
            )
            for worker in self._embed_workers
        ]
        for thread in self._ocr_threads + self._embed_threads:
            thread.start()
        self._log.info("Worker supervisor started")

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._ocr_threads + self._embed_threads:
            thread.join(timeout=2.0)
        self._log.info("Worker supervisor stopped")

    def flush(self) -> None:
        """Flush any pending worker queues (best-effort)."""

    def notify_ocr_observation(self, observation_id: str) -> None:
        # Placeholder for future queue-based workers.
        _ = observation_id
