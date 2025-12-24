"""Global hotkey handling."""

from __future__ import annotations

from pynput import keyboard

from ..logging_utils import get_logger


class GlobalHotkey:
    def __init__(self, hotkey: str, callback) -> None:
        self._hotkey = keyboard.HotKey(keyboard.HotKey.parse(hotkey), callback)
        self._listener = keyboard.Listener(
            on_press=self._hotkey.press, on_release=self._hotkey.release
        )
        self._log = get_logger("hotkey")

    def start(self) -> None:
        self._log.info("Registering hotkey %s", self._hotkey)
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()


def on_hotkey(hotkey: str, callback) -> GlobalHotkey:
    hotkey_listener = GlobalHotkey(hotkey, callback)
    hotkey_listener.start()
    return hotkey_listener
