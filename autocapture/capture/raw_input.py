"""Windows Raw Input listener for HID activity and hotkeys."""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..logging_utils import get_logger
from ..tracking.types import InputVectorEvent

if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong


WM_INPUT = 0x00FF
WM_HOTKEY = 0x0312
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003

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
        self._hwnd: Optional[int] = None
        self._wndproc = None
        self._hotkey_id = 1
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
        if self._hwnd:
            if self._on_hotkey:
                self._unregister_hotkey(self._hwnd)
            ctypes.windll.user32.PostMessageW(self._hwnd, WM_CLOSE, 0, 0)
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
        hwnd = self._create_message_window()
        if not hwnd:
            self._log.error("Failed to create Raw Input message window")
            return
        self._hwnd = hwnd
        if not self._register_raw_input_devices(hwnd):
            self._log.error("Failed to register Raw Input devices")
            return
        if self._on_hotkey and not self._register_hotkey(hwnd):
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
        wndclass = WNDCLASS()
        wndclass.lpfnWndProc = ctypes.cast(wndproc, ctypes.c_void_p)
        wndclass.lpszClassName = class_name
        wndclass.hInstance = hinstance

        if not user32.RegisterClassW(ctypes.byref(wndclass)):
            error = ctypes.GetLastError()
            if error != 1410:  # class already exists
                self._log.error("RegisterClass failed: {}", error)
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
        if self._on_hotkey is None:
            return True
        if self._hotkey_registrar:
            return bool(
                self._hotkey_registrar(hwnd, self._hotkey.modifiers, self._hotkey.vk)
            )
        return bool(
            ctypes.windll.user32.RegisterHotKey(
                hwnd, self._hotkey_id, self._hotkey.modifiers, self._hotkey.vk
            )
        )

    def _unregister_hotkey(self, hwnd: int) -> bool:
        if self._hotkey_unregistrar:
            return bool(self._hotkey_unregistrar(hwnd, self._hotkey_id))
        return bool(ctypes.windll.user32.UnregisterHotKey(hwnd, self._hotkey_id))

    def _handle_wm_input(self, lparam: int) -> None:
        self._mark_activity()
        if not self._on_input_event:
            return
        try:
            data_size = ctypes.c_uint(0)
            ctypes.windll.user32.GetRawInputData(
                lparam,
                RID_INPUT,
                None,
                ctypes.byref(data_size),
                ctypes.sizeof(RAWINPUTHEADER),
            )
            if data_size.value == 0:
                return
            buffer = ctypes.create_string_buffer(data_size.value)
            read = ctypes.windll.user32.GetRawInputData(
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
                    "left_clicks": 1
                    if mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN
                    else 0,
                    "right_clicks": 1
                    if mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN
                    else 0,
                    "middle_clicks": 1
                    if mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN
                    else 0,
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
