"""Windows capture backend leveraging DXGI via :mod:`mss`.

The long-term plan for the project is to ship a dedicated Windows Graphics
Capture (DirectX) binding. To unblock early adopters we provide a performant
Python implementation that relies on :mod:`mss` for frame acquisition and Win32
APIs (via :mod:`ctypes`) for metadata. The implementation honours the
"skip fullscreen" requirement and returns frames enriched with foreground
process/window information so downstream components can reason about activity.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from typing import Any, Optional

from PIL import Image


@dataclass
class RawFrame:
    """Container for a captured frame and its metadata."""

    image: Image.Image
    foreground_process: str
    foreground_window: str
    monitor_id: str
    is_fullscreen: bool


class DirectXSession:  # pragma: no cover - requires Windows runtime to test
    """Capture frames from the primary monitor when running on Windows."""

    def __init__(self, include_cursor: bool = True) -> None:
        self.include_cursor = include_cursor
        self._sct: Optional[Any] = None
        self._monitor: Optional[dict[str, int]] = None
        self._psutil: Optional[Any] = None
        self._ctypes: Optional[Any] = None
        self._wintypes: Optional[Any] = None
        self._user32: Optional[Any] = None
        self._get_foreground_window: Optional[Any] = None
        self._get_window_text: Optional[Any] = None
        self._get_window_rect: Optional[Any] = None
        self._get_window_thread_process_id: Optional[Any] = None

    def initialize(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError(
                "DirectXSession is only available on Windows. Current platform: %s"
                % sys.platform
            )

        self._ctypes = importlib.import_module("ctypes")
        self._wintypes = importlib.import_module("ctypes.wintypes")

        try:
            mss_module = importlib.import_module("mss")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Optional dependency 'mss' is required for Windows capture. "
                "Install it via `pip install autocapture[windows]` or add 'mss' to your environment."
            ) from exc

        try:
            self._psutil = importlib.import_module("psutil")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Optional dependency 'psutil' is required for Windows capture. "
                "Install it via `pip install psutil`."
            ) from exc

        self._user32 = self._ctypes.windll.user32  # type: ignore[attr-defined]

        self._get_foreground_window = self._user32.GetForegroundWindow
        self._get_foreground_window.restype = self._wintypes.HWND

        self._get_window_text = self._user32.GetWindowTextW
        self._get_window_text.argtypes = (
            self._wintypes.HWND,
            self._wintypes.LPWSTR,
            self._ctypes.c_int,
        )

        self._get_window_rect = self._user32.GetWindowRect
        self._get_window_rect.argtypes = (
            self._wintypes.HWND,
            self._ctypes.POINTER(self._wintypes.RECT),
        )

        self._get_window_thread_process_id = self._user32.GetWindowThreadProcessId
        self._get_window_thread_process_id.argtypes = (
            self._wintypes.HWND,
            self._ctypes.POINTER(self._wintypes.DWORD),
        )

        self._sct = mss_module.mss(with_cursor=self.include_cursor)
        primary = self._sct.monitors[1]
        self._monitor = {
            "left": primary["left"],
            "top": primary["top"],
            "width": primary["width"],
            "height": primary["height"],
        }

    def capture(self) -> Optional[RawFrame]:
        if not self._sct or not self._monitor or not self._ctypes:
            raise RuntimeError("DirectXSession.capture() called before initialize().")

        if self._get_foreground_window is None:
            raise RuntimeError("DirectXSession not initialised correctly (foreground).")
        foreground_hwnd = self._get_foreground_window()
        if not foreground_hwnd:
            return None

        if self._get_window_text is None:
            raise RuntimeError(
                "DirectXSession not initialised correctly (window text)."
            )

        buffer = self._ctypes.create_unicode_buffer(512)
        self._get_window_text(foreground_hwnd, buffer, 512)
        window_title = buffer.value

        if self._get_window_thread_process_id is None:
            raise RuntimeError("DirectXSession not initialised correctly (pid).")

        pid = self._wintypes.DWORD()
        self._get_window_thread_process_id(foreground_hwnd, self._ctypes.byref(pid))
        process_name = ""
        if pid.value and self._psutil is not None:
            try:
                process = self._psutil.Process(pid.value)
                process_name = process.name()
            except Exception:  # pragma: no cover - defensive on Windows
                process_name = str(pid.value)

        if self._get_window_rect is None:
            raise RuntimeError("DirectXSession not initialised correctly (rect).")

        rect = self._wintypes.RECT()
        if not self._get_window_rect(foreground_hwnd, self._ctypes.byref(rect)):
            raise self._ctypes.WinError()

        monitor_left = self._monitor["left"]
        monitor_top = self._monitor["top"]
        monitor_right = monitor_left + self._monitor["width"]
        monitor_bottom = monitor_top + self._monitor["height"]

        is_fullscreen = (
            rect.left <= monitor_left
            and rect.top <= monitor_top
            and rect.right >= monitor_right
            and rect.bottom >= monitor_bottom
        )

        if is_fullscreen:
            return None

        frame = self._sct.grab(self._monitor)
        image = Image.frombytes("RGB", frame.size, frame.rgb)

        monitor_id = f"{self._monitor['left']}x{self._monitor['top']}"

        return RawFrame(
            image=image,
            foreground_process=process_name,
            foreground_window=window_title,
            monitor_id=monitor_id,
            is_fullscreen=is_fullscreen,
        )

    def close(self) -> None:
        if self._sct:
            self._sct.close()
            self._sct = None
