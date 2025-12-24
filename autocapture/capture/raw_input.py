"""Windows Raw Input listener for HID activity and hotkeys."""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..logging_utils import get_logger

if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong


WM_INPUT = 0x00FF
WM_HOTKEY = 0x0312
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
RIDEV_INPUTSINK = 0x00000100

MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
VK_SPACE = 0x20

HWND_MESSAGE = -3


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", ctypes.c_ushort),
        ("usUsage", ctypes.c_ushort),
        ("dwFlags", ctypes.c_uint),
        ("hwndTarget", ctypes.c_void_p),
    ]


class WNDCLASS(ctypes.Structure):
    _fields_ = [
        ("style", ctypes.c_uint),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", ctypes.c_void_p),
        ("hIcon", ctypes.c_void_p),
        ("hCursor", ctypes.c_void_p),
        ("hbrBackground", ctypes.c_void_p),
        ("lpszMenuName", ctypes.c_wchar_p),
        ("lpszClassName", ctypes.c_wchar_p),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint),
        ("wParam", ULONG_PTR),
        ("lParam", ULONG_PTR),
        ("time", ctypes.c_uint),
        ("pt_x", ctypes.c_long),
        ("pt_y", ctypes.c_long),
    ]


@dataclass(slots=True)
class HotkeyConfig:
    modifiers: int = MOD_CONTROL | MOD_SHIFT
    vk: int = VK_SPACE


class RawInputListener:
    """Listen for Windows HID activity using Raw Input messages."""

    def __init__(
        self,
        idle_grace_ms: int,
        on_activity: Optional[Callable[[], None]] = None,
        on_hotkey: Optional[Callable[[], None]] = None,
        hotkey: HotkeyConfig | None = None,
    ) -> None:
        self._idle_grace_ms = idle_grace_ms
        self._on_activity = on_activity
        self._on_hotkey = on_hotkey
        self._hotkey = hotkey or HotkeyConfig()
        self._log = get_logger("raw_input")
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._hwnd: Optional[int] = None
        self._wndproc = None
        self._hotkey_id = 1
        self.last_input_ts = self._now_ms()
        self.active_until_ts = self.last_input_ts

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._log.info("Raw input listener started")

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        if self._hwnd:
            ctypes.windll.user32.UnregisterHotKey(self._hwnd, self._hotkey_id)
            ctypes.windll.user32.PostMessageW(self._hwnd, WM_CLOSE, 0, 0)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._log.info("Raw input listener stopped")

    def _run_loop(self) -> None:  # pragma: no cover - Windows message loop
        hwnd = self._create_message_window()
        if not hwnd:
            self._log.error("Failed to create Raw Input message window")
            return
        self._hwnd = hwnd
        if not self._register_raw_input_devices(hwnd):
            self._log.error("Failed to register Raw Input devices")
            return
        if not self._register_hotkey(hwnd):
            self._log.warning("Failed to register hotkey")

        msg = MSG()
        while self._running.is_set():
            result = ctypes.windll.user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result == 0:
                break
            if result == -1:
                self._log.error("GetMessage failed in Raw Input loop")
                break
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))

        if self._hwnd:
            ctypes.windll.user32.DestroyWindow(self._hwnd)
            self._hwnd = None

    def _create_message_window(self) -> int:
        user32 = ctypes.windll.user32
        hinstance = ctypes.windll.kernel32.GetModuleHandleW(None)
        class_name = "AutocaptureRawInputWindow"

        WNDPROCTYPE = ctypes.WINFUNCTYPE(
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_uint,
            ULONG_PTR,
            ULONG_PTR,
        )

        @WNDPROCTYPE
        def wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                self._mark_activity()
                return 0
            if msg == WM_HOTKEY:
                if self._on_hotkey:
                    try:
                        self._on_hotkey()
                    except Exception as exc:  # pragma: no cover
                        self._log.exception("Hotkey callback failed: %s", exc)
                return 0
            if msg == WM_CLOSE:
                user32.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc = wndproc
        wndclass = WNDCLASS()
        wndclass.lpfnWndProc = ctypes.cast(wndproc, ctypes.c_void_p)
        wndclass.lpszClassName = class_name
        wndclass.hInstance = hinstance

        if not user32.RegisterClassW(ctypes.byref(wndclass)):
            error = ctypes.GetLastError()
            if error != 1410:  # class already exists
                self._log.error("RegisterClass failed: %s", error)
                return 0

        hwnd = user32.CreateWindowExW(
            0,
            class_name,
            class_name,
            0,
            0,
            0,
            0,
            0,
            HWND_MESSAGE,
            None,
            hinstance,
            None,
        )
        return int(hwnd)

    def _register_raw_input_devices(self, hwnd: int) -> bool:
        devices = (RAWINPUTDEVICE * 2)()
        devices[0].usUsagePage = 0x01
        devices[0].usUsage = 0x02  # mouse
        devices[0].dwFlags = RIDEV_INPUTSINK
        devices[0].hwndTarget = hwnd

        devices[1].usUsagePage = 0x01
        devices[1].usUsage = 0x06  # keyboard
        devices[1].dwFlags = RIDEV_INPUTSINK
        devices[1].hwndTarget = hwnd

        return bool(
            ctypes.windll.user32.RegisterRawInputDevices(
                devices, len(devices), ctypes.sizeof(RAWINPUTDEVICE)
            )
        )

    def _register_hotkey(self, hwnd: int) -> bool:
        return bool(
            ctypes.windll.user32.RegisterHotKey(
                hwnd, self._hotkey_id, self._hotkey.modifiers, self._hotkey.vk
            )
        )

    def _mark_activity(self) -> None:
        now = self._now_ms()
        self.last_input_ts = now
        self.active_until_ts = now + self._idle_grace_ms
        if self._on_activity:
            try:
                self._on_activity()
            except Exception as exc:  # pragma: no cover
                self._log.exception("Activity callback failed: %s", exc)

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)
