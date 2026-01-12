"""Windows foreground window helpers."""

from __future__ import annotations

import sys

from ..logging_utils import get_logger
from .types import ForegroundContext


if sys.platform != "win32":

    def get_foreground_context() -> ForegroundContext | None:
        return None

    def is_fullscreen_window(hwnd: int) -> bool:
        return False

else:
    import ctypes
    from ctypes import wintypes

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

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    def _get_window_bounds(hwnd: int) -> RECT | None:
        rect = RECT()
        try:
            dwmapi = ctypes.windll.dwmapi
            DWMWA_EXTENDED_FRAME_BOUNDS = 9
            if (
                dwmapi.DwmGetWindowAttribute(
                    hwnd,
                    DWMWA_EXTENDED_FRAME_BOUNDS,
                    ctypes.byref(rect),
                    ctypes.sizeof(rect),
                )
                == 0
            ):
                return rect
        except Exception:
            pass
        if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return rect
        return None

    def is_fullscreen_window(hwnd: int) -> bool:
        if not hwnd:
            return False
        user32 = ctypes.windll.user32
        hmonitor = user32.MonitorFromWindow(hwnd, 2)
        if not hmonitor:
            return False
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        if not user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            return False
        rect = _get_window_bounds(hwnd)
        if not rect:
            return False
        monitor = info.rcMonitor
        monitor_width = monitor.right - monitor.left
        monitor_height = monitor.bottom - monitor.top
        if monitor_width <= 0 or monitor_height <= 0:
            return False
        intersect_w = max(0, min(rect.right, monitor.right) - max(rect.left, monitor.left))
        intersect_h = max(0, min(rect.bottom, monitor.bottom) - max(rect.top, monitor.top))
        coverage = (intersect_w * intersect_h) / float(monitor_width * monitor_height)
        tolerance = 8
        within_bounds = (
            abs(rect.left - monitor.left) <= tolerance
            and abs(rect.top - monitor.top) <= tolerance
            and abs((rect.right - rect.left) - monitor_width) <= tolerance
            and abs((rect.bottom - rect.top) - monitor_height) <= tolerance
        )
        return coverage >= 0.98 or within_bounds
