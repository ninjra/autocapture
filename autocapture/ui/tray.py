"""System tray application with search popup."""

from __future__ import annotations

import webbrowser
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import configure_logging, load_config
from ..config import AppConfig
from ..logging_utils import get_logger
from ..runtime import AppRuntime
from .popup import SearchPopup
from .raw_input import on_hotkey


class TrayApp(QtCore.QObject):
    def __init__(self, config: AppConfig, log_dir: Path, runtime: AppRuntime) -> None:
        super().__init__()
        self._config = config
        self._log_dir = log_dir
        self._runtime = runtime
        self._log = get_logger("tray")
        self._paused = False
        self._hotkey = None

        self._tray = QtWidgets.QSystemTrayIcon(self._default_icon())
        self._menu = QtWidgets.QMenu()
        self._search_action = QtGui.QAction("Search", self._menu)
        self._dashboard_action = QtGui.QAction("Open Dashboard", self._menu)
        self._pause_action = QtGui.QAction("Pause Capture", self._menu)
        self._logs_action = QtGui.QAction("Open Logs Folder", self._menu)
        self._quit_action = QtGui.QAction("Quit", self._menu)

        self._menu.addAction(self._search_action)
        self._menu.addAction(self._dashboard_action)
        self._menu.addSeparator()
        self._menu.addAction(self._pause_action)
        self._menu.addAction(self._logs_action)
        self._menu.addSeparator()
        self._menu.addAction(self._quit_action)
        self._tray.setContextMenu(self._menu)

        self._search_action.triggered.connect(self.show_popup)
        self._dashboard_action.triggered.connect(self.open_dashboard)
        self._pause_action.triggered.connect(self.toggle_pause)
        self._logs_action.triggered.connect(self.open_logs)
        self._quit_action.triggered.connect(self._quit)

        api_url = f"http://{config.api.bind_host}:{config.api.port}"
        self._popup = SearchPopup(api_url)
        self._tray.activated.connect(self._on_tray_activated)

    def start(self) -> None:
        self._tray.show()
        self._hotkey = on_hotkey("<ctrl>+<shift>+<space>", self.toggle_popup)
        self._log.info("Tray started")

    def stop(self) -> None:
        if self._hotkey:
            self._hotkey.stop()

    def toggle_popup(self) -> None:
        if self._popup.isVisible():
            self._popup.hide()
        else:
            self._popup.show_popup()

    def show_popup(self) -> None:
        self._popup.show_popup()

    def open_dashboard(self) -> None:
        url = f"http://{self._config.api.bind_host}:{self._config.api.port}/"
        webbrowser.open(url)

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self._pause_action.setText("Resume Capture" if self._paused else "Pause Capture")
        if self._paused:
            self._runtime.pause_capture()
        else:
            self._runtime.resume_capture()

    def open_logs(self) -> None:
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(self._log_dir.resolve()))
        )

    def _on_tray_activated(self, reason: QtWidgets.QSystemTrayIcon.ActivationReason) -> None:
        if reason == QtWidgets.QSystemTrayIcon.Trigger:
            self.toggle_popup()

    def _quit(self) -> None:
        try:
            self._runtime.stop()
        finally:
            QtWidgets.QApplication.quit()

    @staticmethod
    def _default_icon() -> QtGui.QIcon:
        return QtGui.QIcon.fromTheme("system-search")


def run_tray(config_path: Path, log_dir: Path) -> None:
    configure_logging(log_dir)
    config = load_config(config_path)
    app = QtWidgets.QApplication([])
    runtime = AppRuntime(config)
    runtime.start()
    tray = TrayApp(config, log_dir, runtime)
    tray.start()

    exit_code = app.exec()
    tray.stop()
    runtime.stop()
    raise SystemExit(exit_code)
