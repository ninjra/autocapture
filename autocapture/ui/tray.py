"""System tray application with search popup."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import IO, Optional

import httpx
from PySide6 import QtCore, QtGui, QtWidgets

from .. import configure_logging, load_config
from ..config import AppConfig
from ..logging_utils import get_logger
from ..runtime import AppRuntime
from .popup import SearchPopup


class TrayApp(QtCore.QObject):
    def __init__(
        self,
        config: AppConfig,
        log_dir: Path,
        runtime: AppRuntime,
        api_process: subprocess.Popen | None,
        api_log: Optional[IO[str]],
    ) -> None:
        super().__init__()
        self._config = config
        self._log_dir = log_dir
        self._runtime = runtime
        self._api_process = api_process
        self._api_log = api_log
        self._log = get_logger("tray")
        self._paused = False

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
        self._runtime.set_hotkey_callback(self.toggle_popup)
        self._log.info("Tray started")

    def stop(self) -> None:
        self._runtime.set_hotkey_callback(None)
        self._stop_api_process()

    def toggle_popup(self) -> None:
        if self._popup.isVisible():
            self._popup.hide()
        else:
            self._popup.show_popup()

    def show_popup(self) -> None:
        self._popup.show_popup()

    def open_dashboard(self) -> None:
        self._wait_for_api_ready()
        url = f"http://{self._config.api.bind_host}:{self._config.api.port}/"
        webbrowser.open(url)

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self._pause_action.setText(
            "Resume Capture" if self._paused else "Pause Capture"
        )
        if self._paused:
            self._runtime.pause_capture()
        else:
            self._runtime.resume_capture()

    def open_logs(self) -> None:
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(self._log_dir.resolve()))
        )

    def _on_tray_activated(
        self, reason: QtWidgets.QSystemTrayIcon.ActivationReason
    ) -> None:
        if reason == QtWidgets.QSystemTrayIcon.Trigger:
            self.toggle_popup()

    def _wait_for_api_ready(self) -> None:
        url = f"http://{self._config.api.bind_host}:{self._config.api.port}/health"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                response = httpx.get(url, timeout=1.0)
                if response.status_code == 200:
                    return
            except Exception:
                time.sleep(0.2)
        self._log.warning("API health check timed out; opening dashboard anyway")

    def _stop_api_process(self) -> None:
        proc = self._api_process
        if not proc:
            return
        self._log.info("Stopping API server process")
        if sys.platform == "win32":
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break is not None:
                proc.send_signal(ctrl_break)
            else:
                proc.terminate()
        else:
            proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        self._api_process = None
        if self._api_log:
            self._api_log.close()
            self._api_log = None

    def _quit(self) -> None:
        try:
            self._runtime.stop()
        finally:
            self._stop_api_process()
            QtWidgets.QApplication.quit()

    @staticmethod
    def _default_icon() -> QtGui.QIcon:
        return QtGui.QIcon.fromTheme("system-search")


def _start_api_process(
    config_path: Path, log_dir: Path
) -> tuple[subprocess.Popen, IO[str]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    api_log_path = log_dir / "api.log"
    api_log = api_log_path.open("a", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "autocapture.main",
        "--config",
        str(config_path),
        "api",
    ]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        cmd,
        stdout=api_log,
        stderr=api_log,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    return proc, api_log


def run_tray(config_path: Path, log_dir: Path) -> None:
    configure_logging(log_dir)
    config = load_config(config_path)
    app = QtWidgets.QApplication([])
    runtime = AppRuntime(config)
    runtime.start()
    api_process, api_log = _start_api_process(config_path, log_dir)
    tray = TrayApp(config, log_dir, runtime, api_process, api_log)
    tray.start()

    exit_code = app.exec()
    tray.stop()
    runtime.stop()
    raise SystemExit(exit_code)
