"""Windows global hotkey manager for overlay tracker."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Callable

from ....config import OverlayHotkeySpec, OverlayTrackerHotkeysConfig

# RegisterHotKey reference:
# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-registerhotkey
# WM_HOTKEY reference:
# https://learn.microsoft.com/en-us/windows/win32/inputdev/wm-hotkey

if sys.platform != "win32":

    class HotkeyManager:  # pragma: no cover - non-Windows stub
        def __init__(self, *args, **kwargs) -> None:
            self._status = {}

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        @property
        def status(self) -> dict:
            return self._status

else:
    import ctypes
    from ctypes import wintypes

    MOD_ALT = 0x0001
    MOD_CONTROL = 0x0002
    MOD_SHIFT = 0x0004
    MOD_WIN = 0x0008
    MOD_NOREPEAT = 0x4000

    VK_F1 = 0x70
    VK_F24 = 0x87
    VK_SPACE = 0x20
    VK_TAB = 0x09
    VK_ESCAPE = 0x1B
    VK_RETURN = 0x0D
    VK_BACK = 0x08
    VK_DELETE = 0x2E

    WM_HOTKEY = 0x0312
    WM_QUIT = 0x0012

    @dataclass(slots=True)
    class HotkeyStatus:
        requested: str
        registered: str | None
        ok: bool
        error: str | None

    class HotkeyManager:
        def __init__(
            self,
            config: OverlayTrackerHotkeysConfig,
            callbacks: dict[str, Callable[[], None]],
        ) -> None:
            self._config = config
            self._callbacks = callbacks
            self._status: dict[str, HotkeyStatus] = {}
            self._thread: threading.Thread | None = None
            self._stop = threading.Event()
            self._thread_id: int | None = None
            self._user32 = ctypes.windll.user32
            self._user32.RegisterHotKey.argtypes = [
                wintypes.HWND,
                wintypes.INT,
                wintypes.UINT,
                wintypes.UINT,
            ]
            self._user32.RegisterHotKey.restype = ctypes.c_bool
            self._user32.UnregisterHotKey.argtypes = [wintypes.HWND, wintypes.INT]
            self._user32.UnregisterHotKey.restype = ctypes.c_bool
            self._user32.GetMessageW.argtypes = [
                ctypes.POINTER(wintypes.MSG),
                wintypes.HWND,
                wintypes.UINT,
                wintypes.UINT,
            ]
            self._user32.GetMessageW.restype = ctypes.c_int
            self._user32.PostThreadMessageW.argtypes = [
                wintypes.DWORD,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]
            self._user32.PostThreadMessageW.restype = ctypes.c_bool
            self._hotkey_map: dict[int, str] = {}

        @property
        def status(self) -> dict[str, HotkeyStatus]:
            return self._status

        def start(self) -> None:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def stop(self) -> None:
            self._stop.set()
            if self._thread_id:
                self._user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            if self._thread:
                self._thread.join(timeout=2.0)

        def _run(self) -> None:
            self._thread_id = (
                threading.get_native_id() if hasattr(threading, "get_native_id") else None
            )
            self._register_hotkeys()
            msg = wintypes.MSG()
            while not self._stop.is_set():
                result = self._user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if result <= 0:
                    break
                if msg.message == WM_HOTKEY:
                    action = self._hotkey_map.get(int(msg.wParam))
                    if action:
                        callback = self._callbacks.get(action)
                        if callback:
                            callback()
            self._unregister_hotkeys()

        def _register_hotkeys(self) -> None:
            action_specs = {
                "toggle_overlay": self._config.toggle_overlay,
                "interactive_mode": self._config.interactive_mode,
                "project_cycle": self._config.project_cycle,
                "toggle_running": self._config.toggle_running,
                "rename": self._config.rename,
                "snooze": self._config.snooze,
            }
            hotkey_id = 1
            for action, spec in action_specs.items():
                requested = _format_spec(spec)
                registered = None
                ok = False
                error = None
                for modifiers, vk in _fallback_variants(spec):
                    if self._user32.RegisterHotKey(None, hotkey_id, modifiers, vk):
                        registered = _format_vk(modifiers, vk)
                        ok = True
                        self._hotkey_map[hotkey_id] = action
                        hotkey_id += 1
                        break
                if not ok:
                    error = "register_failed"
                self._status[action] = HotkeyStatus(
                    requested=requested,
                    registered=registered,
                    ok=ok,
                    error=error,
                )
            # Track unused IDs to keep UnregisterHotKey safe
            self._last_hotkey_id = hotkey_id

        def _unregister_hotkeys(self) -> None:
            if not hasattr(self, "_last_hotkey_id"):
                return
            for hotkey_id in range(1, self._last_hotkey_id):
                self._user32.UnregisterHotKey(None, hotkey_id)

    def _fallback_variants(spec: OverlayHotkeySpec) -> list[tuple[int, int]]:
        modifiers = _parse_modifiers(spec.modifiers)
        vk = _parse_vk(spec.key)
        if vk is None:
            return []
        variants = [(modifiers | MOD_NOREPEAT, vk)]
        if not modifiers & MOD_SHIFT:
            variants.append((modifiers | MOD_SHIFT | MOD_NOREPEAT, vk))
        if not modifiers & MOD_ALT:
            variants.append((modifiers | MOD_ALT | MOD_NOREPEAT, vk))
        variants.append((modifiers | MOD_SHIFT | MOD_ALT | MOD_NOREPEAT, vk))
        if vk != VK_F24:
            variants.append((modifiers | MOD_NOREPEAT, VK_F24))
        # Deduplicate while preserving order.
        seen = set()
        unique: list[tuple[int, int]] = []
        for entry in variants:
            if entry in seen:
                continue
            seen.add(entry)
            unique.append(entry)
        return unique

    def _parse_modifiers(mods: list[str]) -> int:
        value = 0
        for mod in mods:
            key = mod.strip().lower()
            if key == "ctrl" or key == "control":
                value |= MOD_CONTROL
            elif key == "shift":
                value |= MOD_SHIFT
            elif key == "alt":
                value |= MOD_ALT
            elif key == "win" or key == "windows":
                value |= MOD_WIN
        return value

    def _parse_vk(key: str) -> int | None:
        label = key.strip().upper()
        if not label:
            return None
        if len(label) == 1 and label.isalpha():
            return ord(label)
        if len(label) == 1 and label.isdigit():
            return ord(label)
        if label.startswith("F") and label[1:].isdigit():
            idx = int(label[1:])
            if 1 <= idx <= 24:
                return VK_F1 + (idx - 1)
        mapping = {
            "SPACE": VK_SPACE,
            "TAB": VK_TAB,
            "ESC": VK_ESCAPE,
            "ESCAPE": VK_ESCAPE,
            "ENTER": VK_RETURN,
            "RETURN": VK_RETURN,
            "BACKSPACE": VK_BACK,
            "DELETE": VK_DELETE,
        }
        return mapping.get(label)

    def _format_spec(spec: OverlayHotkeySpec) -> str:
        mods = "+".join(mod.upper() for mod in spec.modifiers) if spec.modifiers else ""
        return f"{mods}+{spec.key.upper()}" if mods else spec.key.upper()

    def _format_vk(modifiers: int, vk: int) -> str:
        parts: list[str] = []
        if modifiers & MOD_CONTROL:
            parts.append("CTRL")
        if modifiers & MOD_SHIFT:
            parts.append("SHIFT")
        if modifiers & MOD_ALT:
            parts.append("ALT")
        if modifiers & MOD_WIN:
            parts.append("WIN")
        key = f"VK_{vk:02X}"
        return "+".join([*parts, key]) if parts else key
