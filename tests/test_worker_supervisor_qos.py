from __future__ import annotations

import threading
import time

from autocapture.config import AppConfig, DatabaseConfig, RuntimeQosProfile
from autocapture.runtime_governor import RuntimeMode
from autocapture.storage.database import DatabaseManager
from autocapture.worker.supervisor import WorkerSupervisor


class DummyWorker:
    def __init__(self, counter: list[str], label: str) -> None:
        self._counter = counter
        self._label = label

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        while True:
            if stop_event and stop_event.is_set():
                return
            self._counter.append(self._label)
            time.sleep(0.01)


class FakeRuntime:
    def __init__(self, active: RuntimeQosProfile, idle: RuntimeQosProfile) -> None:
        self._active = active
        self._idle = idle
        self.current_mode = RuntimeMode.ACTIVE_INTERACTIVE
        self._callbacks = []

    def subscribe(self, cb):
        self._callbacks.append(cb)

    def unsubscribe(self, cb):
        self._callbacks = [fn for fn in self._callbacks if fn is not cb]

    def qos_profile(self, mode=None):
        mode = mode or self.current_mode
        return self._idle if mode == RuntimeMode.IDLE_DRAIN else self._active

    def set_mode(self, mode: RuntimeMode) -> None:
        self.current_mode = mode
        for cb in list(self._callbacks):
            cb(mode)


def _alive_count(slots) -> int:
    return sum(1 for slot in slots if slot.thread and slot.thread.is_alive())


def test_qos_adjusts_worker_counts(tmp_path) -> None:
    config = AppConfig(database=DatabaseConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    config.worker.ocr_workers = 3
    config.worker.embed_workers = 2
    config.worker.agent_workers = 1
    db = DatabaseManager(config.database)

    active = RuntimeQosProfile(
        ocr_workers=1,
        embed_workers=0,
        agent_workers=0,
        vision_extract=False,
        ui_grounding=False,
        cpu_priority="below_normal",
    )
    idle = RuntimeQosProfile(
        ocr_workers=3,
        embed_workers=2,
        agent_workers=1,
        vision_extract=True,
        ui_grounding=True,
        cpu_priority="normal",
    )
    runtime = FakeRuntime(active, idle)

    counter: list[str] = []
    ocr_workers = [DummyWorker(counter, "ocr") for _ in range(3)]
    embed_workers = [DummyWorker(counter, "embed") for _ in range(2)]
    agent_workers = [DummyWorker(counter, "agent")]

    supervisor = WorkerSupervisor(
        config=config,
        db_manager=db,
        ocr_workers=ocr_workers,
        embed_workers=embed_workers,
        agent_workers=agent_workers,
        runtime_governor=runtime,  # type: ignore[arg-type]
    )
    supervisor.start()
    time.sleep(0.05)
    assert _alive_count(supervisor._ocr_slots) == 1
    assert _alive_count(supervisor._embed_slots) == 0

    runtime.set_mode(RuntimeMode.IDLE_DRAIN)
    time.sleep(0.05)
    assert _alive_count(supervisor._ocr_slots) == 3
    assert _alive_count(supervisor._embed_slots) == 2

    supervisor.stop()
