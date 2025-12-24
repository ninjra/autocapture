"""MSS-based capture backend used as fallback."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ...logging_utils import get_logger
from .monitor_utils import MonitorInfo


class MssBackend:
    """Capture all monitors using mss."""

    def __init__(self) -> None:
        self._log = get_logger("mss")
        import mss  # type: ignore

        self._mss = mss.mss()
        self._monitors: list[MonitorInfo] = []
        for index, monitor in enumerate(self._mss.monitors[1:], start=1):
            self._monitors.append(
                MonitorInfo(
                    id=str(index - 1),
                    left=monitor["left"],
                    top=monitor["top"],
                    width=monitor["width"],
                    height=monitor["height"],
                )
            )

    @property
    def monitors(self) -> list[MonitorInfo]:
        return list(self._monitors)

    def grab_all(self) -> Dict[str, np.ndarray]:
        frames: Dict[str, np.ndarray] = {}
        for monitor, info in zip(self._mss.monitors[1:], self._monitors, strict=True):
            try:
                shot = self._mss.grab(monitor)
                frame = np.asarray(shot)[:, :, :3]
                frames[info.id] = frame
            except Exception as exc:  # pragma: no cover - depends on mss
                self._log.warning("MSS capture failed for monitor %s: %s", info.id, exc)
        return frames
