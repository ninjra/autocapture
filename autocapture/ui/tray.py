"""System tray application with search popup."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
import webbrowser
import datetime as dt
from collections import deque
from pathlib import Path
from typing import IO, Optional

import httpx
from PySide6 import QtCore, QtGui, QtWidgets

from .. import configure_logging, load_config
from ..config import AppConfig
from ..logging_utils import get_logger
from ..runtime import AppRuntime
from ..runtime_env import ProfileName
from ..tracking.win_foreground import get_foreground_context
from ..win32.startup import (
    build_startup_command,
    disable_startup,
    enable_startup,
    is_startup_enabled,
)
from .api_supervisor import ApiSupervisor
from .popup import SearchPopup
from .restart_limiter import RestartLimiter


class TrayApp(QtCore.QObject):
    def __init__(
        self,
        config: AppConfig,
        config_path: Path,
        log_dir: Path,
        runtime: AppRuntime,
        api_process: subprocess.Popen | None,
        api_log: Optional[IO[str]],
    ) -> None:
        super().__init__()
        self._config = config
        self._config_path = config_path
        self._log_dir = log_dir
        self._runtime = runtime
        self._api_process = api_process
        self._api_log = api_log
        self._log = get_logger("tray")
        self._paused = False
        self._api_restart_limiter = RestartLimiter(
            window_s=60.0, max_attempts=3, cooldown_s=120.0
        )
        self._api_restart_notice_at = 0.0

        self._tray = QtWidgets.QSystemTrayIcon(self._default_icon())
        self._menu = QtWidgets.QMenu()
        self._search_action = QtGui.QAction("Search", self._menu)
        self._dashboard_action = QtGui.QAction("Open Dashboard", self._menu)
        self._pause_action = QtGui.QAction("Pause now", self._menu)
        self._snooze_menu = QtWidgets.QMenu("Snooze capture", self._menu)
        self._snooze_5_action = QtGui.QAction("5 minutes", self._menu)
        self._snooze_15_action = QtGui.QAction("15 minutes", self._menu)
        self._snooze_60_action = QtGui.QAction("60 minutes", self._menu)
        self._snooze_resume_action = QtGui.QAction("Resume now", self._menu)
        self._perf_menu = QtWidgets.QMenu("Performance mode", self._menu)
        self._perf_auto_action = QtGui.QAction("Auto (balanced)", self._menu)
        self._perf_max_action = QtGui.QAction("Max capture", self._menu)
        self._perf_low_action = QtGui.QAction("Low impact", self._menu)
        self._perf_group = QtGui.QActionGroup(self._menu)
        self._perf_group.setExclusive(True)
        self._exclude_app_action = QtGui.QAction("Exclude Current App", self._menu)
        self._startup_action = QtGui.QAction("Start on login", self._menu)
        self._delete_15_action = QtGui.QAction("Delete last 15 minutes", self._menu)
        self._delete_24_action = QtGui.QAction("Delete last 24 hours", self._menu)
        self._delete_all_action = QtGui.QAction("Delete everything", self._menu)
        self._logs_action = QtGui.QAction("Open Logs Folder", self._menu)
        self._quit_action = QtGui.QAction("Quit", self._menu)

        self._menu.addAction(self._search_action)
        self._menu.addAction(self._dashboard_action)
        self._menu.addSeparator()
        self._menu.addAction(self._pause_action)
        self._menu.addMenu(self._snooze_menu)
        self._menu.addMenu(self._perf_menu)
        self._menu.addAction(self._exclude_app_action)
        self._menu.addSeparator()
        self._menu.addAction(self._startup_action)
        self._menu.addSeparator()
        self._menu.addAction(self._delete_15_action)
        self._menu.addAction(self._delete_24_action)
        self._menu.addAction(self._delete_all_action)
        self._menu.addSeparator()
        self._menu.addAction(self._logs_action)
        self._menu.addSeparator()
        self._menu.addAction(self._quit_action)
        self._tray.setContextMenu(self._menu)

        self._search_action.triggered.connect(self.show_popup)
        self._dashboard_action.triggered.connect(self.open_dashboard)
        self._pause_action.triggered.connect(self.toggle_pause)
        self._snooze_5_action.triggered.connect(lambda: self._snooze_for(5))
        self._snooze_15_action.triggered.connect(lambda: self._snooze_for(15))
        self._snooze_60_action.triggered.connect(lambda: self._snooze_for(60))
        self._snooze_resume_action.triggered.connect(self.resume_capture)
        self._perf_auto_action.triggered.connect(lambda: self._set_perf_profile(None))
        self._perf_max_action.triggered.connect(
            lambda: self._set_perf_profile(ProfileName.FOREGROUND)
        )
        self._perf_low_action.triggered.connect(
            lambda: self._set_perf_profile(ProfileName.IDLE)
        )
        self._exclude_app_action.triggered.connect(self.exclude_current_app)
        self._startup_action.triggered.connect(self.toggle_startup)
        self._delete_15_action.triggered.connect(lambda: self.delete_range(minutes=15))
        self._delete_24_action.triggered.connect(lambda: self.delete_range(hours=24))
        self._delete_all_action.triggered.connect(self.delete_all)
        self._logs_action.triggered.connect(self.open_logs)
        self._quit_action.triggered.connect(self._quit)

        self._snooze_menu.addAction(self._snooze_5_action)
        self._snooze_menu.addAction(self._snooze_15_action)
        self._snooze_menu.addAction(self._snooze_60_action)
        self._snooze_menu.addSeparator()
        self._snooze_menu.addAction(self._snooze_resume_action)

        for action in (self._perf_auto_action, self._perf_max_action, self._perf_low_action):
            action.setCheckable(True)
            self._perf_group.addAction(action)
            self._perf_menu.addAction(action)

        self._startup_action.setCheckable(True)
        self._startup_action.setChecked(is_startup_enabled())

        self._api_url = f"http://{config.api.bind_host}:{config.api.port}"
        self._popup = SearchPopup(self._api_url, config=self._config.ui.search_popup)
        self._tray.activated.connect(self._on_tray_activated)
        self._health_timer = QtCore.QTimer(self)
        self._health_timer.setInterval(5000)
        self._health_timer.timeout.connect(self._tick_supervisor)
        self._supervisor = ApiSupervisor(self._api_is_healthy, self._restart_api)
        self._sync_pause_state()
        self._sync_profile_state()

    def start(self) -> None:
        self._tray.show()
        self._runtime.set_hotkey_callback(self.toggle_popup)
        self._health_timer.start()
        self._log.info("Tray started")

    def stop(self) -> None:
        self._runtime.set_hotkey_callback(None)
        self._health_timer.stop()
        self._stop_api_process()

    def toggle_popup(self) -> None:
        if self._popup.isVisible():
            self._popup.hide()
        else:
            self.show_popup()

    def show_popup(self) -> None:
        if not self._ensure_api_ready("Search popup"):
            return
        self._popup.show_popup()

    def open_dashboard(self) -> None:
        if not self._ensure_api_ready("Dashboard"):
            return
        token = self._request_unlock_token()
        url = f"http://{self._config.api.bind_host}:{self._config.api.port}/"
        if token:
            url = f"{url}?unlock={token}"
        webbrowser.open(url)

    def toggle_pause(self) -> None:
        if self._paused:
            self._runtime.resume_capture()
        else:
            self._runtime.pause_capture()
        self._sync_pause_state()

    def resume_capture(self) -> None:
        self._runtime.resume_capture()
        self._sync_pause_state()

    def _snooze_for(self, minutes: int) -> None:
        self._runtime.snooze_capture(minutes)
        self._sync_pause_state()

    def exclude_current_app(self) -> None:
        ctx = get_foreground_context()
        process_name = ctx.process_name if ctx else None
        if not process_name:
            self._notify("Exclude App", "Unable to detect the current app.")
            return
        if self._runtime.add_excluded_process(process_name):
            self._notify("Exclude App", f"Added {process_name} to excluded apps.")
        else:
            self._notify("Exclude App", f"{process_name} is already excluded.")

    def toggle_startup(self) -> None:
        if sys.platform != "win32":
            self._startup_action.setChecked(False)
            QtWidgets.QMessageBox.information(
                None, "Start on login", "Start on login is only available on Windows."
            )
            return
        command = build_startup_command(self._config_path, self._log_dir)
        if self._startup_action.isChecked():
            enable_startup(command)
            self._notify("Start on login", "Autocapture will start on login.")
        else:
            disable_startup()
            self._notify("Start on login", "Autocapture will not start on login.")

    def delete_range(self, *, minutes: int = 0, hours: int = 0) -> None:
        duration = dt.timedelta(minutes=minutes, hours=hours)
        if duration.total_seconds() <= 0:
            return
        label = f"{int(duration.total_seconds() // 60)} minutes"
        if hours:
            label = f"{hours} hours"
        if not self._confirm_action(f"Delete the last {label}?"):
            return
        now = dt.datetime.now(dt.timezone.utc)
        start = now - duration
        self._delete_range_request(start, now)

    def delete_all(self) -> None:
        if not self._confirm_action("Delete all captured history? This cannot be undone."):
            return
        self._delete_range_request(None, None, endpoint="/api/delete_all")

    def open_logs(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self._log_dir.resolve())))

    def _on_tray_activated(self, reason: QtWidgets.QSystemTrayIcon.ActivationReason) -> None:
        if reason == QtWidgets.QSystemTrayIcon.Trigger:
            self.toggle_popup()

    def _wait_for_api_ready(self) -> bool:
        url = f"{self._api_url}/health"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                response = httpx.get(url, timeout=1.0)
                if response.status_code == 200:
                    return True
            except Exception:
                time.sleep(0.2)
        self._log.warning("API health check timed out; opening dashboard anyway")
        return False

    def _api_is_healthy(self) -> bool:
        try:
            response = httpx.get(f"{self._api_url}/healthz/deep", timeout=1.5)
            return response.status_code == 200
        except Exception:
            return False

    def _tick_supervisor(self) -> None:
        self._sync_pause_state()
        self._sync_profile_state()
        if self._api_process and self._api_process.poll() is not None:
            exit_code = self._api_process.returncode
            self._log.warning("API process exited (code={}); awaiting supervisor restart.", exit_code)
            self._stop_api_process()
        state = self._supervisor.tick()
        self._tray.setToolTip(self._build_status_tooltip(state.status))

    def _restart_api(self) -> None:
        decision = self._api_restart_limiter.attempt()
        if not decision.allowed:
            now = time.monotonic()
            if decision.reason == "loop":
                self._log.error(
                    "API restart loop detected; pausing restarts for {}s.",
                    int(decision.cooldown_remaining_s),
                )
                self._log_api_tail()
                self._notify(
                    "Autocapture",
                    "API keeps crashing. Restarts paused to prevent loops. Check logs.",
                )
            elif now - self._api_restart_notice_at > 5.0:
                self._api_restart_notice_at = now
                self._log.warning(
                    "API restart paused (cooldown {}s remaining).",
                    int(decision.cooldown_remaining_s),
                )
            return
        self._log.warning("Restarting API process")
        self._stop_api_process()
        self._api_process, self._api_log = _start_api_process(self._config_path, self._log_dir)

    def _request_unlock_token(self) -> str | None:
        try:
            response = httpx.post(f"{self._api_url}/api/unlock", timeout=10.0)
            if response.status_code != 200:
                return None
            data = response.json()
            token = data.get("token")
            return token or None
        except Exception as exc:
            self._log.warning("Unlock request failed: {}", exc)
            return None

    def _delete_range_request(
        self,
        start: dt.datetime | None,
        end: dt.datetime | None,
        *,
        endpoint: str = "/api/delete_range",
    ) -> None:
        token = self._request_unlock_token()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = None
        if start and end:
            payload = {"start_utc": start.isoformat(), "end_utc": end.isoformat()}
        try:
            response = httpx.post(
                f"{self._api_url}{endpoint}",
                json=payload,
                headers=headers,
                timeout=20.0,
            )
            if response.status_code != 200:
                raise RuntimeError(response.text)
            data = response.json()
            QtWidgets.QMessageBox.information(
                None,
                "Deletion complete",
                (
                    f"Deleted {data.get('deleted_events', 0)} events and "
                    f"{data.get('deleted_captures', 0)} captures."
                ),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(None, "Deletion failed", f"{exc}")

    def _confirm_action(self, message: str) -> bool:
        result = QtWidgets.QMessageBox.question(
            None,
            "Confirm action",
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        return result == QtWidgets.QMessageBox.Yes

    def _stop_api_process(self) -> None:
        proc = self._api_process
        if not proc:
            return
        self._log.info("Stopping API server process")
        if proc.poll() is None:
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

    def _ensure_api_ready(self, action_label: str) -> bool:
        if self._api_process is None or self._api_process.poll() is not None:
            self._supervisor.tick()
        if not self._wait_for_api_ready():
            self._notify(
                action_label,
                "Autocapture is still starting up. Please try again in a moment.",
            )
            return False
        return True

    def _sync_pause_state(self) -> None:
        paused = bool(self._config.privacy.paused)
        self._paused = paused
        self._pause_action.setText("Resume Capture" if paused else "Pause now")

    def _sync_profile_state(self) -> None:
        state = self._runtime.profile_state()
        override = state.get("override")
        if override == ProfileName.FOREGROUND.value:
            self._perf_max_action.setChecked(True)
        elif override == ProfileName.IDLE.value:
            self._perf_low_action.setChecked(True)
        else:
            self._perf_auto_action.setChecked(True)

    def _set_perf_profile(self, profile: ProfileName | None) -> None:
        self._runtime.set_performance_profile(profile)
        if profile == ProfileName.FOREGROUND:
            label = "Max capture"
        elif profile == ProfileName.IDLE:
            label = "Low impact"
        else:
            label = "Auto (balanced)"
        self._notify("Performance mode", f"{label} enabled.")
        self._sync_profile_state()

    def _build_status_tooltip(self, api_status: str) -> str:
        now = dt.datetime.now(dt.timezone.utc)
        snooze_until = self._config.privacy.snooze_until_utc
        if snooze_until and snooze_until.tzinfo is None:
            snooze_until = snooze_until.replace(tzinfo=dt.timezone.utc)
        if self._config.privacy.paused:
            if snooze_until and snooze_until > now:
                local_time = snooze_until.astimezone().strftime("%Y-%m-%d %H:%M")
                status = f"Snoozed until {local_time}"
            else:
                status = "Paused"
        else:
            status = "Capturing"
        if api_status == "backoff":
            return f"Autocapture: {status} (API recovering)"
        if api_status == "restarting":
            return f"Autocapture: {status} (API restarting)"
        return f"Autocapture: {status}"

    def _notify(self, title: str, message: str) -> None:
        if self._tray.supportsMessages():
            self._tray.showMessage(title, message)

    def _log_api_tail(self, lines: int = 120) -> None:
        for name in ("api.log", "autocapture.log"):
            path = self._log_dir / name
            if not path.exists():
                continue
            tail = _tail_text(path, lines=lines)
            if not tail.strip():
                continue
            self._log.error("===== {} (tail) =====\n{}", name, tail)

    def _quit(self) -> None:
        try:
            self._runtime.stop()
        finally:
            self._stop_api_process()
            QtWidgets.QApplication.quit()

    @staticmethod
    def _default_icon() -> QtGui.QIcon:
        return QtGui.QIcon.fromTheme("system-search")


def _start_api_process(config_path: Path, log_dir: Path) -> tuple[subprocess.Popen, IO[str]]:
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


def _tail_text(path: Path, *, lines: int = 120) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            tail = deque(handle, maxlen=lines)
    except Exception as exc:  # pragma: no cover - best-effort logging
        return f"(failed to read {path.name}: {exc})"
    return "".join(tail)


def run_tray(config_path: Path, log_dir: Path) -> None:
    configure_logging(log_dir)
    config = load_config(config_path)
    app = QtWidgets.QApplication([])
    runtime = AppRuntime(config)
    runtime.start()
    api_process, api_log = _start_api_process(config_path, log_dir)
    tray = TrayApp(config, config_path, log_dir, runtime, api_process, api_log)
    tray.start()

    exit_code = app.exec()
    tray.stop()
    runtime.stop()
    raise SystemExit(exit_code)
