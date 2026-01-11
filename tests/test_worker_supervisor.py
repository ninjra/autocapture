from __future__ import annotations

import threading
import time

import pytest

from autocapture.config import AppConfig, DatabaseConfig
from autocapture.worker.supervisor import WorkerSupervisor


class CrashOnceWorker:
    def __init__(self) -> None:
        self.started = 0
        self.crashed = threading.Event()
        self.restarted = threading.Event()

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        self.started += 1
        if self.started == 1:
            self.crashed.set()
            raise RuntimeError("boom")
        self.restarted.set()
        while stop_event and not stop_event.is_set():
            time.sleep(0.01)


@pytest.mark.filterwarnings(
    "ignore:Exception in thread:pytest.PytestUnhandledThreadExceptionWarning"
)
def test_worker_supervisor_restarts_dead_worker(tmp_path) -> None:
    config = AppConfig(
        database=DatabaseConfig(url="sqlite:///:memory:"),
        capture={"data_dir": tmp_path, "staging_dir": tmp_path / "staging"},
        embed={"text_model": "local-test"},
        worker={"watchdog_interval_s": 0.05},
    )
    crash_worker = CrashOnceWorker()
    supervisor = WorkerSupervisor(
        config=config,
        ocr_workers=[crash_worker],
        embed_workers=[],
    )

    supervisor.start()
    assert crash_worker.crashed.wait(0.5)
    assert crash_worker.restarted.wait(0.5)
    supervisor.stop()
