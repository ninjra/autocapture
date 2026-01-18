"""Windows input activity polling via GetLastInputInfo."""

from __future__ import annotations

import sys
import threading
import time

from ...schemas import OverlayCollectorContext, OverlayCollectorEvent
from ...clock import Clock

# GetLastInputInfo reference:
# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getlastinputinfo
# LASTINPUTINFO reference:
# https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-lastinputinfo

if sys.platform != "win32":

    class InputActivityCollector:  # pragma: no cover - non-Windows stub
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

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    class InputActivityCollector:
        def __init__(
            self,
            *,
            clock: Clock,
            on_event,
            poll_ms: int,
        ) -> None:
            self._clock = clock
            self._on_event = on_event
            self._poll_s = poll_ms / 1000.0
            self._thread: threading.Thread | None = None
            self._stop = threading.Event()
            self._last_tick: int | None = None
            self._status = "idle"
            self._user32 = ctypes.windll.user32
            self._user32.GetLastInputInfo.argtypes = [ctypes.POINTER(LASTINPUTINFO)]
            self._user32.GetLastInputInfo.restype = ctypes.c_bool

        @property
        def status(self) -> str:
            return self._status

        def start(self) -> None:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._status = "ok"

        def stop(self) -> None:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=2.0)
            self._status = "stopped"

        def _run(self) -> None:
            while not self._stop.is_set():
                tick = self._get_last_input_tick()
                if tick is not None and tick != self._last_tick:
                    self._last_tick = tick
                    event = OverlayCollectorEvent(
                        event_type="input_activity",
                        ts_utc=self._clock.now(),
                        context=OverlayCollectorContext(process_name="unknown"),
                        collector="get_last_input_info",
                        metadata={"tick": tick},
                    )
                    try:
                        self._on_event(event)
                    except Exception:
                        pass
                time.sleep(self._poll_s)

        def _get_last_input_tick(self) -> int | None:
            info = LASTINPUTINFO()
            info.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if not self._user32.GetLastInputInfo(ctypes.byref(info)):
                return None
            return int(info.dwTime)
