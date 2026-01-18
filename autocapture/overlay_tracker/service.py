"""Overlay tracker service wiring (Windows-only collectors + store)."""

from __future__ import annotations

import datetime as dt
import sys
from typing import Any, Callable

from ..config import OverlayTrackerConfig
from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from .clock import SystemClock
from .engine import OverlayCommand, OverlayTrackerEngine
from .store import OverlayTrackerStore
from .collectors.windows.foreground import ForegroundCollector
from .collectors.windows.input_activity import InputActivityCollector
from .collectors.windows.hotkeys import HotkeyManager


class OverlayTrackerService:
    """Lifecycle wrapper for the overlay tracker module."""

    name = "overlay_tracker"

    def __init__(
        self,
        config: OverlayTrackerConfig,
        db: DatabaseManager,
        *,
        sanitize: bool = False,
    ) -> None:
        self._config = config
        self._db = db
        self._log = get_logger("overlay_tracker")
        self._started = False
        self._sanitize = sanitize
        self._clock = SystemClock()
        self._store = OverlayTrackerStore(db, self._clock)
        self._engine = OverlayTrackerEngine(config, self._store, self._clock)
        self._foreground: ForegroundCollector | None = None
        self._input: InputActivityCollector | None = None
        self._hotkeys: HotkeyManager | None = None
        self._ui = None

    @property
    def enabled(self) -> bool:
        return bool(
            self._config.enabled
            and sys.platform == "win32"
            and "windows" in [p.lower() for p in self._config.platforms]
        )

    def start(self) -> None:
        if not self.enabled:
            self._log.info("Overlay tracker disabled")
            return
        if self._started:
            return
        self._started = True
        self._log.info("Overlay tracker starting")
        self._engine.start()
        if self._config.collectors.foreground_enabled:
            self._foreground = ForegroundCollector(
                clock=self._clock,
                on_event=self._engine.submit_event,
                fallback_poll_ms=self._config.collectors.fallback_foreground_poll_ms,
                max_title_len=self._config.policy.max_window_title_length,
            )
            self._foreground.start()
        if self._config.collectors.input_enabled:
            self._input = InputActivityCollector(
                clock=self._clock,
                on_event=self._engine.submit_event,
                poll_ms=self._config.collectors.input_poll_ms,
            )
            self._input.start()
        self._hotkeys = HotkeyManager(self._config.hotkeys, self._hotkey_callbacks())
        self._hotkeys.start()
        self._start_ui()

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self._hotkeys:
            self._hotkeys.stop()
        if self._input:
            self._input.stop()
        if self._foreground:
            self._foreground.stop()
        if self._ui:
            self._ui.stop()
            self._ui = None
        self._engine.stop()
        self._log.info("Overlay tracker stopped")

    def health(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "started": self._started,
            "current_item_id": self._engine.current_item_id,
            "foreground_status": self._foreground.status if self._foreground else None,
            "input_status": self._input.status if self._input else None,
            "hotkeys": self._hotkeys.status if self._hotkeys else None,
            "engine": self._engine.health(),
            "ui_enabled": bool(self._ui),
        }

    def _hotkey_callbacks(self) -> dict[str, Callable[[], None]]:
        return {
            "toggle_overlay": self._toggle_overlay,
            "interactive_mode": self._interactive_mode,
            "project_cycle": lambda: self._engine.submit_command(
                OverlayCommand(action="cycle_project", payload={})
            ),
            "toggle_running": lambda: self._engine.submit_command(
                OverlayCommand(action="toggle_running", payload={})
            ),
            "rename": self._rename_current,
            "snooze": self._snooze_current,
        }

    def _toggle_overlay(self) -> None:
        if not self._ui:
            return None
        self._ui.invoke(self._ui.toggle_visible)

    def _interactive_mode(self) -> None:
        if not self._ui:
            return None
        self._ui.invoke(lambda: self._ui.set_interactive(True))

    def _rename_current(self) -> None:
        if not self._ui:
            return None
        item_id = self._engine.current_item_id
        if not item_id:
            return None
        self._ui.invoke(lambda: _prompt_rename(item_id, self._engine, self._ui))

    def _snooze_current(self) -> None:
        item_id = self._engine.current_item_id
        if not item_id:
            return None
        minutes = self._config.hotkeys.snooze_minutes
        duration = minutes[0] if minutes else 15
        until = self._clock.now() + dt.timedelta(minutes=duration)
        self._engine.submit_command(OverlayCommand(action="snooze", payload={"until": until}))

    def _start_ui(self) -> None:
        if not self._config.ui.enabled:
            return
        if self._ui:
            return
        try:
            from PySide6 import QtWidgets
            from .ui.overlay_ui import OverlayUiController
        except Exception as exc:  # pragma: no cover - optional UI dependency
            self._log.warning("Overlay UI unavailable: {}", exc)
            return
        app = QtWidgets.QApplication.instance()
        if app is None:
            self._log.warning("Overlay UI skipped (no Qt application)")
            return
        self._ui = OverlayUiController(
            self._config,
            self._store,
            self._clock,
            sanitize=self._sanitize,
        )
        self._ui.start()


def _prompt_rename(item_id: int, engine: OverlayTrackerEngine, ui) -> None:
    from PySide6 import QtWidgets

    text, ok = QtWidgets.QInputDialog.getText(
        ui.window,
        "Rename Item",
        "Display name:",
    )
    if ok and text:
        engine.submit_command(OverlayCommand(action="rename", payload={"name": text}))
