"""Utilities for diagnosing unexpected subprocess or interpreter spawns."""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

_INSTALLED = False


def _write_log(log_path: Path, heading: str, payload: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("[" + timestamp + f"] {heading}\n")
        handle.write(payload)
        if not payload.endswith("\n"):
            handle.write("\n")
        handle.write("\n")


def install_spawn_debugging(log_path: str | os.PathLike[str] | None = None) -> None:
    """Log every attempt to spawn Python interpreters or subprocesses.

    When Windows services unexpectedly launch ``python -m autocapture.main`` using
    the system interpreter, this helper can be enabled to capture the call site
    and stack trace that triggered the spawn. Logging is only enabled once per
    process and is guarded so non-Windows platforms simply return.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    _INSTALLED = True

    if log_path is None:
        log_path = os.environ.get("AUTOCAPTURE_DEBUG_SPAWN_LOG", "spawn-debug.log")

    log_file = Path(log_path).expanduser().resolve()
    header = (
        f"pid={os.getpid()} parent={os.getppid()} "
        f"executable={sys.executable!r} argv={sys.argv!r}"
    )
    _write_log(log_file, "spawn-debugging-initialised", header)

    def log_stack(event: str, extra: str) -> None:
        stack = "".join(traceback.format_stack())
        payload = f"{extra}\nStack:\n{stack}"
        _write_log(log_file, event, payload)

    # Patch subprocess to capture direct invocations.
    original_popen = subprocess.Popen

    def logged_popen(*args: Any, **kwargs: Any):  # type: ignore[override]
        cmd = args[0] if args else kwargs.get("args")
        log_stack("subprocess.Popen", f"args={cmd!r}")
        return original_popen(*args, **kwargs)

    subprocess.Popen = logged_popen  # type: ignore[assignment]

    original_run = getattr(subprocess, "run", None)
    if callable(original_run):

        def logged_run(*args: Any, **kwargs: Any):  # type: ignore[override]
            cmd = args[0] if args else kwargs.get("args")
            log_stack("subprocess.run", f"args={cmd!r}")
            return original_run(*args, **kwargs)

        subprocess.run = logged_run  # type: ignore[assignment]

    if sys.platform == "win32":  # pragma: no cover - windows specific
        try:
            import multiprocessing.popen_spawn_win32 as popen_win32
        except (ImportError, ModuleNotFoundError):
            return

        original_init = popen_win32.Popen.__init__

        def logged_init(self: Any, *args: Any, **kwargs: Any) -> None:
            log_stack(
                "multiprocessing.Popen",
                f"process_obj={getattr(args[0], '__class__', type(args[0]))!r}",
            )
            original_init(self, *args, **kwargs)

        popen_win32.Popen.__init__ = logged_init  # type: ignore[assignment]
