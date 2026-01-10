"""Windows Raw Input listener for HID activity and hotkeys."""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ctypes import wintypes

from ..logging_utils import get_logger
from ..tracking.types import InputVectorEvent

if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong

WINFUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)

LRESULT = ULONG_PTR
WPARAM = ULONG_PTR
LPARAM = ULONG_PTR
HWND = wintypes.HWND
UINT = wintypes.UINT


WM_INPUT = 0x00FF
WM_HOTKEY = 0x0312
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_QUIT = 0x0012
RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
PM_REMOVE = 0x0001

RIM_TYPEMOUSE = 0
RIM_TYPEKEYBOARD = 1

RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
RI_MOUSE_RIGHT_BUTTON_DOWN = 0x0004
RI_MOUSE_MIDDLE_BUTTON_DOWN = 0x0010
RI_MOUSE_WHEEL = 0x0400

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


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
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
        ("hIconSm", ctypes.c_void_p),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", HWND),
        ("message", UINT),
        ("wParam", WPARAM),
        ("lParam", LPARAM),
        ("time", ctypes.c_uint),
        ("pt_x", ctypes.c_long),
        ("pt_y", ctypes.c_long),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", ctypes.c_uint),
        ("dwSize", ctypes.c_uint),
        ("hDevice", ctypes.c_void_p),
        ("wParam", ULONG_PTR),
    ]


class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", ctypes.c_ushort),
        ("usButtonFlags", ctypes.c_ushort),
        ("usButtonData", ctypes.c_ushort),
        ("ulRawButtons", ctypes.c_uint),
        ("lLastX", ctypes.c_long),
        ("lLastY", ctypes.c_long),
        ("ulExtraInformation", ctypes.c_uint),
    ]


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", ctypes.c_ushort),
        ("Flags", ctypes.c_ushort),
        ("Reserved", ctypes.c_ushort),
        ("VKey", ctypes.c_ushort),
        ("Message", ctypes.c_uint),
        ("ExtraInformation", ctypes.c_uint),
    ]


class RAWINPUTDATA(ctypes.Union):
    _fields_ = [("mouse", RAWMOUSE), ("keyboard", RAWKEYBOARD)]


class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("data", RAWINPUTDATA)]


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


class Win32Api:
    def __init__(self) -> None:
        if sys.platform != "win32":  # pragma: no cover - Windows-only
            raise RuntimeError("Win32 API is only available on Windows")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure()

    def _configure(self) -> None:
        self.user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
        self.user32.RegisterClassExW.restype = ctypes.c_ushort
        self.user32.CreateWindowExW.argtypes = [
            ctypes.c_uint,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.user32.CreateWindowExW.restype = ctypes.c_void_p
        self.user32.DestroyWindow.argtypes = [ctypes.c_void_p]
        self.user32.DestroyWindow.restype = ctypes.c_bool
        self.user32.DefWindowProcW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            WPARAM,
            LPARAM,
        ]
        self.user32.DefWindowProcW.restype = LRESULT
        self.user32.GetMessageW.argtypes = [
            ctypes.POINTER(MSG),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self.user32.GetMessageW.restype = ctypes.c_int
        self.user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
        self.user32.TranslateMessage.restype = ctypes.c_bool
        self.user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
        self.user32.DispatchMessageW.restype = LRESULT
        self.user32.PeekMessageW.argtypes = [
            ctypes.POINTER(MSG),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self.user32.PeekMessageW.restype = ctypes.c_bool
        self.user32.PostMessageW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            WPARAM,
            LPARAM,
        ]
        self.user32.PostMessageW.restype = ctypes.c_bool
        self.user32.PostThreadMessageW.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            WPARAM,
            LPARAM,
        ]
        self.user32.PostThreadMessageW.restype = ctypes.c_bool
        self.user32.PostQuitMessage.argtypes = [ctypes.c_int]
        self.user32.PostQuitMessage.restype = None
        self.user32.RegisterRawInputDevices.argtypes = [
            ctypes.POINTER(RAWINPUTDEVICE),
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self.user32.RegisterRawInputDevices.restype = ctypes.c_bool
        self.user32.GetRawInputData.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.c_uint,
        ]
        self.user32.GetRawInputData.restype = ctypes.c_uint
        self.user32.RegisterHotKey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self.user32.RegisterHotKey.restype = ctypes.c_bool
        self.user32.UnregisterHotKey.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.user32.UnregisterHotKey.restype = ctypes.c_bool
        self.user32.GetLastInputInfo.argtypes = [ctypes.POINTER(LASTINPUTINFO)]
        self.user32.GetLastInputInfo.restype = ctypes.c_bool
        self.user32.GetForegroundWindow.argtypes = []
        self.user32.GetForegroundWindow.restype = ctypes.c_void_p
        self.kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        self.kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        self.kernel32.FormatMessageW.argtypes = [
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_wchar_p,
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        self.kernel32.FormatMessageW.restype = ctypes.c_uint

    def last_error(self) -> int:
        return ctypes.get_last_error()

    def format_error(self, code: int) -> str:
        buffer = ctypes.create_unicode_buffer(512)
        flags = 0x00001000 | 0x00000200
        length = self.kernel32.FormatMessageW(
            flags, None, code, 0, buffer, len(buffer), None
        )
        if length:
            return buffer.value.strip()
        return "Unknown error"

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
        on_input_event: Optional[Callable[[InputVectorEvent], None]] = None,
        track_mouse_movement: bool = True,
        mouse_move_sample_ms: int = 50,
        hotkey: HotkeyConfig | None = None,
        hotkey_registrar: Optional[Callable[[int, int, int], bool]] = None,
        hotkey_unregistrar: Optional[Callable[[int, int], bool]] = None,
        win32_api: Win32Api | None = None,
        fallback_poll_s: float = 0.25,
    ) -> None:
        self._idle_grace_ms = idle_grace_ms
        self._on_activity = on_activity
        self._on_hotkey = on_hotkey
        self._on_input_event = on_input_event
        self._track_mouse_movement = track_mouse_movement
        self._mouse_move_sample_ms = mouse_move_sample_ms
        self._hotkey = hotkey or HotkeyConfig()
        self._log = get_logger("raw_input")
        self._hotkey_registrar = hotkey_registrar
        self._hotkey_unregistrar = hotkey_unregistrar
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._thread_id: Optional[int] = None
        self._hwnd: Optional[int] = None
        self._wndproc = None
        self._hotkey_id = 1
        self._fallback_active = False
        self._fallback_poll_s = fallback_poll_s
        self._win32 = win32_api
        if self._win32 is None and sys.platform == "win32":
            self._win32 = Win32Api()
        self.last_input_ts = self._now_ms()
        self.active_until_ts = self.last_input_ts
        self._last_mouse_emit_ts = self.last_input_ts
        self._mouse_move_dx = 0
        self._mouse_move_dy = 0

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
        win32 = self._win32
        if self._hwnd and win32:
            if self._on_hotkey:
                self._unregister_hotkey(self._hwnd)
            win32.user32.PostMessageW(self._hwnd, WM_CLOSE, 0, 0)
        elif self._thread_id and win32:
            win32.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._log.info("Raw input listener stopped")

    def set_hotkey_callback(self, callback: Optional[Callable[[], None]]) -> None:
        previous = self._on_hotkey
        self._on_hotkey = callback
        if not self._hwnd:
            return
        if previous is None and callback is not None:
            if not self._register_hotkey(self._hwnd):
                self._log.warning("Failed to register hotkey")
        elif previous is not None and callback is None:
            self._unregister_hotkey(self._hwnd)

    def _run_loop(self) -> None:  # pragma: no cover - Windows message loop
        if self._win32 is None:
            self._log.warning("Raw input not available on this platform")
            self._running.clear()
            return
        self._thread_id = threading.get_native_id() if hasattr(threading, "get_native_id") else None
        hwnd = self._create_message_window()
        if not hwnd:
            self._log.warning("Failed to create Raw Input message window; enabling fallback")
            self._fallback_active = True
            self._run_fallback_loop()
            return
        self._hwnd = hwnd
        if not self._register_raw_input_devices(hwnd):
            self._log.error("Failed to register Raw Input devices; enabling fallback")
            self._fallback_active = True
            self._run_fallback_loop()
            return
        if self._on_hotkey and not self._register_hotkey(hwnd):
            self._log.warning("Failed to register hotkey")

        msg = MSG()
        win32 = self._win32
        while self._running.is_set() and win32:
            result = win32.user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result == 0:
                break
            if result == -1:
                error = win32.last_error()
                self._log.error(
                    "GetMessage failed in Raw Input loop: {} ({})",
                    error,
                    win32.format_error(error),
                )
                break
            win32.user32.TranslateMessage(ctypes.byref(msg))
            win32.user32.DispatchMessageW(ctypes.byref(msg))

        if self._hwnd and win32:
            win32.user32.DestroyWindow(self._hwnd)
            self._hwnd = None

    def _create_message_window(self) -> int:
        win32 = self._win32
        if win32 is None:
            return 0
        user32 = win32.user32
        hinstance = win32.kernel32.GetModuleHandleW(None)
        class_name = "AutocaptureRawInputWindow"

        WNDPROCTYPE = WINFUNCTYPE(
            LRESULT,
            HWND,
            UINT,
            WPARAM,
            LPARAM,
        )

        @WNDPROCTYPE
        def wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                self._handle_wm_input(lparam)
                return 0
            if msg == WM_HOTKEY:
                if self._on_hotkey:
                    try:
                        self._on_hotkey()
                    except Exception as exc:  # pragma: no cover
                        self._log.exception("Hotkey callback failed: {}", exc)
                return 0
            if msg == WM_CLOSE:
                user32.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc = wndproc
        wndclass = WNDCLASSEXW()
        wndclass.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wndclass.lpfnWndProc = ctypes.cast(wndproc, ctypes.c_void_p)
        wndclass.lpszClassName = class_name
        wndclass.hInstance = hinstance

        if not user32.RegisterClassExW(ctypes.byref(wndclass)):
            error = win32.last_error()
            if error != 1410:  # class already exists
                self._log.error(
                    "RegisterClassExW failed: {} ({})",
                    error,
                    win32.format_error(error),
                )
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
        if not hwnd:
            error = win32.last_error()
            self._log.error(
                "CreateWindowExW failed: {} ({})",
                error,
                win32.format_error(error),
            )
            return 0
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

        win32 = self._win32
        if win32 is None:
            return False
        return bool(
            win32.user32.RegisterRawInputDevices(
                devices, len(devices), ctypes.sizeof(RAWINPUTDEVICE)
            )
        )

    def _register_hotkey(self, hwnd: int) -> bool:
        if self._on_hotkey is None:
            return True
        if self._hotkey_registrar:
            return bool(
                self._hotkey_registrar(hwnd, self._hotkey.modifiers, self._hotkey.vk)
            )
        if not self._win32:
            return False
        return bool(
            self._win32.user32.RegisterHotKey(
                hwnd, self._hotkey_id, self._hotkey.modifiers, self._hotkey.vk
            )
        )

    def _unregister_hotkey(self, hwnd: int) -> bool:
        if self._hotkey_unregistrar:
            return bool(self._hotkey_unregistrar(hwnd, self._hotkey_id))
        if not self._win32:
            return False
        return bool(self._win32.user32.UnregisterHotKey(hwnd, self._hotkey_id))

    def _handle_wm_input(self, lparam: int) -> None:
        self._mark_activity()
        if not self._on_input_event:
            return
        try:
            if not self._win32:
                return
            data_size = ctypes.c_uint(0)
            self._win32.user32.GetRawInputData(
                lparam,
                RID_INPUT,
                None,
                ctypes.byref(data_size),
                ctypes.sizeof(RAWINPUTHEADER),
            )
            if data_size.value == 0:
                return
            buffer = ctypes.create_string_buffer(data_size.value)
            read = self._win32.user32.GetRawInputData(
                lparam,
                RID_INPUT,
                buffer,
                ctypes.byref(data_size),
                ctypes.sizeof(RAWINPUTHEADER),
            )
            if read != data_size.value:
                return
            raw = RAWINPUT.from_buffer_copy(buffer)
            now_ms = self._now_wall_ms()
            if raw.header.dwType == RIM_TYPEKEYBOARD:
                event = InputVectorEvent(
                    ts_ms=now_ms,
                    device="keyboard",
                    mouse={"events": 1},
                )
                self._on_input_event(event)
            elif raw.header.dwType == RIM_TYPEMOUSE:
                mouse = raw.data.mouse
                payload: dict[str, int] = {
                    "left_clicks": (
                        1 if mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN else 0
                    ),
                    "right_clicks": (
                        1 if mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN else 0
                    ),
                    "middle_clicks": (
                        1 if mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN else 0
                    ),
                }
                if mouse.usButtonFlags & RI_MOUSE_WHEEL:
                    payload["wheel_events"] = 1
                    payload["wheel_delta"] = ctypes.c_short(mouse.usButtonData).value
                else:
                    payload["wheel_events"] = 0
                    payload["wheel_delta"] = 0
                emitted_move = False
                if self._track_mouse_movement:
                    self._mouse_move_dx += int(mouse.lLastX)
                    self._mouse_move_dy += int(mouse.lLastY)
                    if now_ms - self._last_mouse_emit_ts >= self._mouse_move_sample_ms:
                        payload["move_dx"] = self._mouse_move_dx
                        payload["move_dy"] = self._mouse_move_dy
                        self._mouse_move_dx = 0
                        self._mouse_move_dy = 0
                        self._last_mouse_emit_ts = now_ms
                        emitted_move = True
                if (
                    payload["left_clicks"]
                    or payload["right_clicks"]
                    or payload["middle_clicks"]
                    or payload["wheel_events"]
                    or emitted_move
                ):
                    event = InputVectorEvent(
                        ts_ms=now_ms, device="mouse", mouse=payload
                    )
                    self._on_input_event(event)
        except Exception as exc:  # pragma: no cover - Windows-only parsing
            self._log.debug("Raw input parse failed: {}", exc)

    def _run_fallback_loop(self) -> None:
        win32 = self._win32
        if win32 is None:
            return
        if self._on_hotkey:
            if not win32.user32.RegisterHotKey(
                None, self._hotkey_id, self._hotkey.modifiers, self._hotkey.vk
            ):
                error = win32.last_error()
                self._log.warning(
                    "Fallback hotkey registration failed: {} ({})",
                    error,
                    win32.format_error(error),
                )
        msg = MSG()
        last_tick = None
        last_window = None
        while self._running.is_set():
            while win32.user32.PeekMessageW(
                ctypes.byref(msg), None, 0, 0, PM_REMOVE
            ):
                if msg.message == WM_HOTKEY and self._on_hotkey:
                    try:
                        self._on_hotkey()
                    except Exception as exc:  # pragma: no cover
                        self._log.exception("Hotkey callback failed: {}", exc)
                if msg.message == WM_QUIT:
                    self._running.clear()
                    break

            tick = self._get_last_input_tick()
            if tick is not None and tick != last_tick:
                last_tick = tick
                self._mark_activity()
            hwnd = win32.user32.GetForegroundWindow()
            if hwnd and hwnd != last_window:
                last_window = hwnd
                self._mark_activity()
            time.sleep(self._fallback_poll_s)

        if self._on_hotkey:
            win32.user32.UnregisterHotKey(None, self._hotkey_id)

    def _get_last_input_tick(self) -> Optional[int]:
        win32 = self._win32
        if win32 is None:
            return None
        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not win32.user32.GetLastInputInfo(ctypes.byref(info)):
            error = win32.last_error()
            self._log.debug(
                "GetLastInputInfo failed: {} ({})",
                error,
                win32.format_error(error),
            )
            return None
        return int(info.dwTime)

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active

    def _mark_activity(self) -> None:
        now = self._now_ms()
        self.last_input_ts = now
        self.active_until_ts = now + self._idle_grace_ms
        if self._on_activity:
            try:
                self._on_activity()
            except Exception as exc:  # pragma: no cover
                self._log.exception("Activity callback failed: {}", exc)

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    @staticmethod
    def _now_wall_ms() -> int:
        return int(time.time() * 1000)


def probe_raw_input() -> dict:
    """Probe Raw Input availability on Windows without starting the listener."""
    if sys.platform != "win32":
        return {"available": False, "error": "non-windows"}
    try:
        win32 = Win32Api()
    except Exception as exc:  # pragma: no cover - Windows-only
        return {"available": False, "error": str(exc)}
    listener = RawInputListener(idle_grace_ms=1000, win32_api=win32)
    hwnd = listener._create_message_window()
    if not hwnd:
        return {"available": False, "error": "window_create_failed"}
    ok = listener._register_raw_input_devices(hwnd)
    win32.user32.DestroyWindow(hwnd)
    if not ok:
        return {"available": False, "error": "register_devices_failed"}
    return {"available": True, "error": None}
