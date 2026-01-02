"""Windows foreground window helpers."""

from __future__ import annotations

import sys

from ..logging_utils import get_logger
from .types import ForegroundContext


if sys.platform != "win32":

    def get_foreground_context() -> ForegroundContext | None:
        return None

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
            log.debug("Failed to read foreground window: %s", exc)
            return None
