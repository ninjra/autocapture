from __future__ import annotations

import threading
import time

import numpy as np

import autocapture.worker.event_worker as event_worker_module
import autocapture.worker.embedding_worker as embedding_worker_module
from autocapture.config import AppConfig, DatabaseConfig, EmbedConfig
from autocapture.storage.database import DatabaseManager
from autocapture.worker.event_worker import EventIngestWorker
from autocapture.worker.embedding_worker import EmbeddingWorker


class FakeOCR:
    def run(self, image: np.ndarray) -> list[tuple[str, float, list[int]]]:
        return [("hello", 0.9, [0, 0, 1, 0, 1, 1, 0, 1])]


class FakeEmbedder:
    def __init__(self) -> None:
        self.model_name = "fake-model"
        self.dim = 3

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_event_worker_loop_survives_exception(monkeypatch) -> None:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:"),
        embed={"text_model": "local-test"},
    )
    db = DatabaseManager(config.database)
    worker = EventIngestWorker(config, db_manager=db, ocr_processor=FakeOCR())

    calls: list[int] = []
    saw_second = threading.Event()
    original_sleep = time.sleep

    def fast_sleep(seconds: float) -> None:
        original_sleep(min(seconds, 0.01))

    def fake_process_batch() -> int:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        saw_second.set()
        return 0

    monkeypatch.setattr(event_worker_module.time, "sleep", fast_sleep)
    monkeypatch.setattr(worker, "process_batch", fake_process_batch)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=worker.run_forever, kwargs={"stop_event": stop_event}, daemon=True
    )
    thread.start()
    assert saw_second.wait(1.0)
    stop_event.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_embedding_worker_loop_survives_exception(monkeypatch) -> None:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:"),
        embed=EmbedConfig(text_model="fake-model"),
    )
    db = DatabaseManager(config.database)
    worker = EmbeddingWorker(config, db_manager=db, embedder=FakeEmbedder())

    calls: list[int] = []
    saw_second = threading.Event()
    original_sleep = time.sleep

    def fast_sleep(seconds: float) -> None:
        original_sleep(min(seconds, 0.01))

    def fake_process_batch() -> int:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        saw_second.set()
        return 0

    monkeypatch.setattr(embedding_worker_module.time, "sleep", fast_sleep)
    monkeypatch.setattr(worker, "process_batch", fake_process_batch)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=worker.run_forever, kwargs={"stop_event": stop_event}, daemon=True
    )
    thread.start()
    assert saw_second.wait(1.0)
    stop_event.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
