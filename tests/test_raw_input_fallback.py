from __future__ import annotations

import ctypes
import time

from autocapture.capture.raw_input import LASTINPUTINFO, RawInputListener


class _FakeUser32:
    def __init__(self) -> None:
        self._last_input = 100

    def RegisterClassExW(self, _cls) -> int:
        return 1

    def CreateWindowExW(self, *_args) -> int:
        return 0

    def RegisterHotKey(self, *_args) -> bool:
        return True

    def UnregisterHotKey(self, *_args) -> bool:
        return True

    def PeekMessageW(self, *_args) -> bool:
        return False

    def GetLastInputInfo(self, info_ptr) -> bool:
        info = ctypes.cast(info_ptr, ctypes.POINTER(LASTINPUTINFO)).contents
        info.dwTime = self._last_input
        self._last_input += 1
        return True

    def GetForegroundWindow(self) -> int:
        return 1

    def PostThreadMessageW(self, *_args) -> bool:
        return True


class _FakeKernel32:
    def GetModuleHandleW(self, _name) -> int:
        return 1

    def FormatMessageW(self, *_args) -> int:
        return 0


class FakeWin32Api:
    def __init__(self) -> None:
        self.user32 = _FakeUser32()
        self.kernel32 = _FakeKernel32()

    def last_error(self) -> int:
        return 5

    def format_error(self, _code: int) -> str:
        return "mock"


def test_raw_input_fallback_activates_on_window_failure() -> None:
    listener = RawInputListener(
        idle_grace_ms=1000,
        on_activity=None,
        on_hotkey=None,
        win32_api=FakeWin32Api(),
        fallback_poll_s=0.01,
    )
    start_ts = listener.last_input_ts
    listener.start()
    time.sleep(0.05)
    listener.stop()
    assert listener.fallback_active is True
    assert listener.last_input_ts > start_ts
