from __future__ import annotations

import threading
import time

from autocapture.llm.governor import LLMGovernor, PressureProvider, PressureSample


class _FixedPressure(PressureProvider):
    def __init__(self, cpu: float | None, gpu: float | None) -> None:
        self._cpu = cpu
        self._gpu = gpu

    def sample(self) -> PressureSample:
        return PressureSample(cpu_util=self._cpu, gpu_util=self._gpu)


def test_llm_governor_adaptive_limits() -> None:
    governor = LLMGovernor(
        enabled=True,
        min_in_flight=1,
        max_in_flight=3,
        low_pressure_threshold=0.2,
        high_pressure_threshold=0.8,
        adjust_interval_s=0.0,
        provider=_FixedPressure(cpu=0.9, gpu=None),
    )
    governor.refresh()
    assert governor.current_limit == 2

    governor = LLMGovernor(
        enabled=True,
        min_in_flight=1,
        max_in_flight=3,
        low_pressure_threshold=0.2,
        high_pressure_threshold=0.8,
        adjust_interval_s=0.0,
        provider=_FixedPressure(cpu=0.1, gpu=None),
    )
    governor.refresh()
    assert governor.current_limit == 3


def test_llm_governor_foreground_priority() -> None:
    governor = LLMGovernor(
        enabled=True,
        min_in_flight=1,
        max_in_flight=1,
        low_pressure_threshold=0.0,
        high_pressure_threshold=1.0,
        adjust_interval_s=100.0,
        provider=_FixedPressure(cpu=None, gpu=None),
    )
    order: list[str] = []
    lock = threading.Lock()

    def _worker(priority: str, started: threading.Event, acquired: threading.Event) -> None:
        started.set()
        with governor.reserve(priority):
            with lock:
                order.append(priority)
            acquired.set()
            time.sleep(0.05)

    started_bg = threading.Event()
    started_fg = threading.Event()
    acquired_bg = threading.Event()
    acquired_fg = threading.Event()

    with governor.reserve("foreground"):
        bg_thread = threading.Thread(
            target=_worker, args=("background", started_bg, acquired_bg), daemon=True
        )
        fg_thread = threading.Thread(
            target=_worker, args=("foreground", started_fg, acquired_fg), daemon=True
        )
        bg_thread.start()
        started_bg.wait(timeout=1.0)
        fg_thread.start()
        started_fg.wait(timeout=1.0)
        time.sleep(0.05)
    acquired_fg.wait(timeout=1.0)
    acquired_bg.wait(timeout=1.0)
    assert order[0] == "foreground"
