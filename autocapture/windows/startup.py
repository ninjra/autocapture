"""Windows scheduled task helpers for autostart."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..logging_utils import get_logger


TASK_NAME = "Autocapture"


def ensure_startup_task(executable: Path) -> bool:
    if sys.platform != "win32":
        return False
    log = get_logger("windows.startup")
    if _task_exists():
        log.info("Startup task already exists.")
        return True
    return create_startup_task(executable)


def create_startup_task(executable: Path) -> bool:
    if sys.platform != "win32":
        return False
    log = get_logger("windows.startup")
    cmd = [
        "schtasks",
        "/Create",
        "/SC",
        "ONLOGON",
        "/TN",
        TASK_NAME,
        "/TR",
        f'"{executable}"',
        "/RL",
        "LIMITED",
        "/F",
    ]
    try:
        subprocess.run(" ".join(cmd), check=True, shell=True)
        log.info("Created startup task %s", TASK_NAME)
        return True
    except subprocess.CalledProcessError as exc:
        log.warning("Failed to create startup task: %s", exc)
        return False


def remove_startup_task() -> bool:
    if sys.platform != "win32":
        return False
    log = get_logger("windows.startup")
    cmd = ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"]
    try:
        subprocess.run(cmd, check=True)
        log.info("Removed startup task %s", TASK_NAME)
        return True
    except subprocess.CalledProcessError as exc:
        log.warning("Failed to remove startup task: %s", exc)
        return False


def _task_exists() -> bool:
    if sys.platform != "win32":
        return False
    try:
        subprocess.run(
            ["schtasks", "/Query", "/TN", TASK_NAME],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False
