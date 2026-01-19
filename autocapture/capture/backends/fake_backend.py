"""Deterministic fake capture backend for tests and CI."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .monitor_utils import MonitorInfo, stable_monitor_id


class FakeBackend:
    """Return a fixed set of frames for deterministic capture tests."""

    def __init__(
        self,
        *,
        frames: Dict[str, np.ndarray] | None = None,
        monitors: list[MonitorInfo] | None = None,
        width: int = 120,
        height: int = 90,
    ) -> None:
        if monitors is None:
            monitor_id = stable_monitor_id(0, 0, width, height)
            monitors = [MonitorInfo(id=monitor_id, left=0, top=0, width=width, height=height)]
        self._monitors = monitors
        if frames is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = 32
            frame[:, :, 1] = 64
            frame[:, :, 2] = 96
            frames = {monitors[0].id: frame}
        self._frames = frames

    @property
    def monitors(self) -> list[MonitorInfo]:
        return list(self._monitors)

    def grab_all(self) -> Dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self._frames.items()}
