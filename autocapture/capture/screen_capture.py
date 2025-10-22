"""Interfaces for GPU-accelerated screen capture."""

from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image


@dataclass(slots=True)
class CaptureFrame:
    timestamp: dt.datetime
    image: Image.Image
    foreground_process: str
    foreground_window: str
    monitor_id: str
    is_fullscreen: bool

    def save(self, path: Path, encoder: str = "webp", quality: int = 90) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path, format=encoder.upper(), quality=quality)
        return path


class ScreenCaptureBackend(abc.ABC):
    """Abstract interface implemented by platform-specific capture backends."""

    @abc.abstractmethod
    def start(self) -> None:  # pragma: no cover - requires Windows API
        """Initialize capture pipelines (e.g., DirectX duplication sessions)."""

    @abc.abstractmethod
    def stop(self) -> None:  # pragma: no cover - requires Windows API
        """Cleanly dispose of capture resources."""

    @abc.abstractmethod
    def capture_once(
        self,
    ) -> Optional[CaptureFrame]:  # pragma: no cover - requires Windows API
        """Grab a single frame if available, otherwise return ``None``."""


class DirectXDesktopDuplicator(ScreenCaptureBackend):  # pragma: no cover - Windows only
    """Thin wrapper around Windows Graphics Capture via DirectX."""

    def __init__(self, include_cursor: bool = True) -> None:
        self.include_cursor = include_cursor
        self._session = None

    def start(self) -> None:
        # Lazy import to avoid platform issues when developing on Linux.
        from .win_directx import DirectXSession  # type: ignore

        self._session = DirectXSession(include_cursor=self.include_cursor)
        self._session.initialize()

    def stop(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

    def capture_once(self) -> Optional[CaptureFrame]:
        if self._session is None:
            return None
        frame = self._session.capture()
        if frame is None:
            return None
        return CaptureFrame(
            timestamp=dt.datetime.utcnow(),
            image=frame.image,
            foreground_process=frame.foreground_process,
            foreground_window=frame.foreground_window,
            monitor_id=frame.monitor_id,
            is_fullscreen=frame.is_fullscreen,
        )
