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
        self._refresh_monitors()

    @property
    def monitors(self) -> list[MonitorInfo]:
        return list(self._monitors)

    def grab_all(self) -> Dict[str, np.ndarray]:
        frames: Dict[str, np.ndarray] = {}
        monitors = self._mss.monitors[1:]
        if len(monitors) != len(self._monitors):
            self._log.warning("Monitor list changed; refreshing MSS monitors.")
            self._refresh_monitors()
            monitors = self._mss.monitors[1:]
        for monitor, info in zip(monitors, self._monitors):
            try:
                shot = self._mss.grab(monitor)
                frame = np.asarray(shot)[:, :, :3][:, :, ::-1]
                frames[info.id] = frame
            except Exception as exc:  # pragma: no cover - depends on mss
                self._log.warning("MSS capture failed for monitor {}: {}", info.id, exc)
        return frames

    def _refresh_monitors(self) -> None:
        self._monitors = []
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
