"""Windows startup registration using HKCU Run key."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..logging_utils import get_logger


_RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"


def build_startup_command(config_path: Path, log_dir: Path | None) -> str:
    args = [sys.executable]
    if getattr(sys, "frozen", False) or getattr(sys, "_MEIPASS", None):
        args.extend(["--config", str(config_path)])
    else:
        args.extend(["-m", "autocapture.main", "--config", str(config_path)])
    if log_dir is not None:
        args.extend(["--log-dir", str(log_dir)])
    args.append("tray")
    return subprocess.list2cmdline(args)


def is_startup_enabled(app_name: str = "Autocapture", *, winreg_module=None) -> bool:
    winreg = _load_winreg(winreg_module)
    if winreg is None:
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY_PATH, 0, winreg.KEY_READ) as key:
            try:
                winreg.QueryValueEx(key, app_name)
                return True
            except FileNotFoundError:
                return False
    except FileNotFoundError:
        return False


def enable_startup(command: str, app_name: str = "Autocapture", *, winreg_module=None) -> None:
    winreg = _load_winreg(winreg_module)
    if winreg is None:
        return
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _RUN_KEY_PATH) as key:
        winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, command)


def disable_startup(app_name: str = "Autocapture", *, winreg_module=None) -> None:
    winreg = _load_winreg(winreg_module)
    if winreg is None:
        return
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _RUN_KEY_PATH, 0, winreg.KEY_SET_VALUE
        ) as key:
            try:
                winreg.DeleteValue(key, app_name)
            except FileNotFoundError:
                return
    except FileNotFoundError:
        return


def _load_winreg(winreg_module=None):
    if winreg_module is not None:
        return winreg_module
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except Exception as exc:  # pragma: no cover - defensive
        get_logger("win32.startup").debug("winreg unavailable: %s", exc)
        return None
    return winreg
