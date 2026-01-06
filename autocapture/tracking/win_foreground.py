"""Windows foreground window helpers."""

from __future__ import annotations

import sys

from ..logging_utils import get_logger
from .types import ForegroundContext


if sys.platform != "win32":

    def get_foreground_context() -> ForegroundContext | None:
        return None

    def is_fullscreen_window(hwnd: int, monitor_bounds: tuple[int, int, int, int]) -> bool:
        return False

else:
    import ctypes

    def get_foreground_context() -> ForegroundContext | None:
        log = get_logger("tracking.foreground")
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None
            length = user32.GetWindowTextLengthW(hwnd)
            title_buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, title_buffer, length + 1)
            pid = ctypes.c_uint()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            import psutil

            process_name = "unknown"
            if pid.value:
                try:
                    process_name = psutil.Process(pid.value).name()
                except psutil.Error:
                    process_name = "unknown"
            return ForegroundContext(
                process_name=process_name,
                window_title=title_buffer.value,
                pid=pid.value or None,
                hwnd=int(hwnd),
            )
        except Exception as exc:  # pragma: no cover - depends on Windows APIs
            log.debug("Failed to read foreground window: {}", exc)
            return None

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    def is_fullscreen_window(hwnd: int, monitor_bounds: tuple[int, int, int, int]) -> bool:
        if not hwnd:
            return False
        rect = RECT()
        if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return False
        left, top, width, height = monitor_bounds
        win_width = rect.right - rect.left
        win_height = rect.bottom - rect.top
        tolerance = 2
        return (
            abs(rect.left - left) <= tolerance
            and abs(rect.top - top) <= tolerance
            and abs(win_width - width) <= tolerance
            and abs(win_height - height) <= tolerance
        )
