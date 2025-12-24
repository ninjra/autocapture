"""Monitor enumeration helpers for capture backends."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Callable


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", ctypes.c_uint),
    ]


@dataclass(slots=True)
class MonitorInfo:
    id: str
    left: int
    top: int
    width: int
    height: int

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.left + self.width and self.top <= y < self.top + self.height


def enumerate_monitors() -> list[MonitorInfo]:
    user32 = ctypes.windll.user32
    monitors: list[MonitorInfo] = []

    MONITORENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_long
    )

    def _callback(hmonitor, _hdc, _rect_ptr, _data):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            rect = info.rcMonitor
            monitors.append(
                MonitorInfo(
                    id=str(len(monitors)),
                    left=rect.left,
                    top=rect.top,
                    width=rect.right - rect.left,
                    height=rect.bottom - rect.top,
                )
            )
        return 1

    callback = MONITORENUMPROC(_callback)
    user32.EnumDisplayMonitors(None, None, callback, 0)
    return monitors
