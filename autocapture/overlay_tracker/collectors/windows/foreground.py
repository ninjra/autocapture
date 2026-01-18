"""Windows foreground window collector for overlay tracker."""

from __future__ import annotations

import sys
import threading
import time

from ...schemas import OverlayCollectorContext, OverlayCollectorEvent
from ...clock import Clock

# NOTE: The Windows WinEvent hook API is documented at:
# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwineventhook

if sys.platform != "win32":

    class ForegroundCollector:  # pragma: no cover - non-Windows stub
        def __init__(self, *args, **kwargs) -> None:
            self._status = "disabled"

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        @property
        def status(self) -> str:
            return self._status

else:
    import ctypes
    from ctypes import wintypes

    import psutil

    # EVENT_SYSTEM_FOREGROUND constant (see SetWinEventHook docs linked above).
    EVENT_SYSTEM_FOREGROUND = 0x0003
    WINEVENT_OUTOFCONTEXT = 0x0000
    WINEVENT_SKIPOWNPROCESS = 0x0002

    HWINEVENTHOOK = wintypes.HANDLE
    DWORD = wintypes.DWORD
    HWND = wintypes.HWND
    LONG = wintypes.LONG
    WINFUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)

    WINEVENTPROC = WINFUNCTYPE(
        None,
        HWINEVENTHOOK,
        DWORD,
        HWND,
        LONG,
        LONG,
        DWORD,
        DWORD,
    )

    class ForegroundHook:
        def __init__(
            self,
            *,
            clock: Clock,
            on_event,
            max_title_len: int,
        ) -> None:
            self._clock = clock
            self._on_event = on_event
            self._max_title_len = max_title_len
            self._thread: threading.Thread | None = None
            self._stop = threading.Event()
            self._hook: HWINEVENTHOOK | None = None
            self._callback = None
            self._status = "idle"
            self._thread_id: int | None = None
            self._ready = threading.Event()

        @property
        def status(self) -> str:
            return self._status

        def start(self) -> bool:
            if self._thread and self._thread.is_alive():
                return True
            self._stop.clear()
            self._ready.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._ready.wait(timeout=0.2)
            return self._status == "ok"

        def stop(self) -> None:
            self._stop.set()
            if self._thread_id:
                ctypes.windll.user32.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)
            if self._thread:
                self._thread.join(timeout=2.0)

        def _run(self) -> None:
            user32 = ctypes.windll.user32
            self._thread_id = (
                threading.get_native_id() if hasattr(threading, "get_native_id") else None
            )
            self._callback = self._build_callback()
            hook = user32.SetWinEventHook(
                EVENT_SYSTEM_FOREGROUND,
                EVENT_SYSTEM_FOREGROUND,
                0,
                self._callback,
                0,
                0,
                WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS,
            )
            if not hook:
                self._status = "failed"
                self._ready.set()
                return
            self._hook = hook
            self._status = "ok"
            self._ready.set()
            msg = wintypes.MSG()
            while not self._stop.is_set():
                if user32.GetMessageW(ctypes.byref(msg), None, 0, 0) <= 0:
                    break
            if self._hook:
                user32.UnhookWinEvent(self._hook)
                self._hook = None

        def _build_callback(self):
            @WINEVENTPROC
            def _handler(_hook, event, hwnd, _id_object, _id_child, _thread_id, _ms):
                if event != EVENT_SYSTEM_FOREGROUND or not hwnd:
                    return
                ctx = _context_from_hwnd(hwnd, self._max_title_len)
                if not ctx:
                    return
                evt = OverlayCollectorEvent(
                    event_type="foreground",
                    ts_utc=self._clock.now(),
                    context=ctx,
                    collector="win_event_hook",
                    metadata={"hwnd": int(hwnd)},
                )
                try:
                    self._on_event(evt)
                except Exception:
                    return

            return _handler

    class ForegroundPoller:
        def __init__(
            self,
            *,
            clock: Clock,
            on_event,
            poll_ms: int,
            max_title_len: int,
        ) -> None:
            self._clock = clock
            self._on_event = on_event
            self._poll_s = poll_ms / 1000.0
            self._max_title_len = max_title_len
            self._thread: threading.Thread | None = None
            self._stop = threading.Event()
            self._last_context: OverlayCollectorContext | None = None

        def start(self) -> None:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def stop(self) -> None:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=2.0)

        def _run(self) -> None:
            user32 = ctypes.windll.user32
            while not self._stop.is_set():
                hwnd = int(user32.GetForegroundWindow() or 0)
                if hwnd:
                    ctx = _context_from_hwnd(hwnd, self._max_title_len)
                    if ctx and (
                        self._last_context is None
                        or ctx.hwnd != self._last_context.hwnd
                        or ctx.process_name != self._last_context.process_name
                        or ctx.window_title != self._last_context.window_title
                    ):
                        self._last_context = ctx
                        evt = OverlayCollectorEvent(
                            event_type="foreground",
                            ts_utc=self._clock.now(),
                            context=ctx,
                            collector="foreground_poll",
                            metadata={"hwnd": int(hwnd)},
                        )
                        try:
                            self._on_event(evt)
                        except Exception:
                            pass
                time.sleep(self._poll_s)

    class ForegroundCollector:
        def __init__(
            self,
            *,
            clock: Clock,
            on_event,
            fallback_poll_ms: int,
            max_title_len: int,
        ) -> None:
            self._hook = ForegroundHook(
                clock=clock,
                on_event=on_event,
                max_title_len=max_title_len,
            )
            self._poller = ForegroundPoller(
                clock=clock,
                on_event=on_event,
                poll_ms=fallback_poll_ms,
                max_title_len=max_title_len,
            )
            self._status = "idle"

        @property
        def status(self) -> str:
            return self._status

        def start(self) -> None:
            ok = self._hook.start()
            if ok:
                self._status = "ok"
                return
            self._poller.start()
            self._status = "fallback"

        def stop(self) -> None:
            self._hook.stop()
            self._poller.stop()
            self._status = "stopped"

    def _context_from_hwnd(hwnd: int, max_title_len: int) -> OverlayCollectorContext | None:
        user32 = ctypes.windll.user32
        length = user32.GetWindowTextLengthW(hwnd)
        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, length + 1)
        pid = ctypes.c_uint()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process_name = "unknown"
        if pid.value:
            try:
                process_name = psutil.Process(pid.value).name()
            except psutil.Error:
                process_name = "unknown"
        title = title_buffer.value
        if max_title_len > 0 and len(title) > max_title_len:
            title = title[:max_title_len]
        return OverlayCollectorContext(
            process_name=process_name,
            window_title=title,
            browser_url=None,
            hwnd=hwnd,
        )
