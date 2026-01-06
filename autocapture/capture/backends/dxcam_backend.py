"""DxCam-based multi-monitor capture backend."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ...logging_utils import get_logger
from .monitor_utils import MonitorInfo, enumerate_monitors


class DxCamBackend:
    """Capture each monitor using a dedicated dxcam camera."""

    def __init__(self) -> None:
        self._log = get_logger("dxcam")
        self._monitors = enumerate_monitors()
        self._cameras: dict[str, object] = {}
        self._disabled: set[str] = set()

        import dxcam  # type: ignore

        for idx, monitor in enumerate(self._monitors):
            try:
                camera = dxcam.create(output_idx=idx)
            except Exception as exc:  # pragma: no cover - depends on dxcam
                self._log.warning("Failed to create dxcam output {}: {}", idx, exc)
                continue
            if camera is None:
                self._log.warning("DxCam returned None for output {}", idx)
                continue
            self._cameras[monitor.id] = camera

    @property
    def monitors(self) -> list[MonitorInfo]:
        return list(self._monitors)

    def grab_all(self) -> Dict[str, np.ndarray]:
        frames: Dict[str, np.ndarray] = {}
        for monitor in self._monitors:
            monitor_id = monitor.id
            if monitor_id in self._disabled:
                continue
            camera = self._cameras.get(monitor_id)
            if camera is None:
                continue
            try:
                frame = camera.grab()
                if frame is None:
                    raise RuntimeError("dxcam returned empty frame")
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                frames[monitor_id] = frame[:, :, ::-1]
            except Exception as exc:  # pragma: no cover - depends on dxcam
                self._log.warning("Disabling dxcam output {}: {}", monitor_id, exc)
                self._disabled.add(monitor_id)
        return frames
