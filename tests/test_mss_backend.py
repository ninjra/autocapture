from __future__ import annotations

import types
import sys

import numpy as np

from autocapture.capture.backends.mss_backend import MssBackend


class DummyShot:
    def __init__(self, width: int, height: int) -> None:
        self._array = np.zeros((height, width, 4), dtype=np.uint8)

    def __array__(self, dtype=None):
        return self._array


class DummyMSS:
    def __init__(self) -> None:
        self.monitors = [
            {"left": 0, "top": 0, "width": 10, "height": 10},
            {"left": 0, "top": 0, "width": 10, "height": 10},
        ]

    def grab(self, _monitor):
        return DummyShot(10, 10)


def test_mss_backend_monitor_mismatch(monkeypatch) -> None:
    dummy_module = types.ModuleType("mss")
    dummy_module.mss = DummyMSS
    monkeypatch.setitem(sys.modules, "mss", dummy_module)

    backend = MssBackend()
    backend._mss.monitors.append({"left": 0, "top": 0, "width": 10, "height": 10})
    frames = backend.grab_all()

    assert frames
