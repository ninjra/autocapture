"""Autocapture package providing a high-performance local recall pipeline."""

from __future__ import annotations

import multiprocessing as _mp
import os
import sys
from pathlib import Path
from typing import Final, Sequence

from .config import AppConfig, load_config
from .logging_utils import configure_logging

if hasattr(_mp, "set_executable"):
    _mp.set_executable(sys.executable)

# Hint for libraries which shell out to ``python`` (e.g. multiprocessing helpers)
# so they inherit the virtualenv interpreter instead of the system install.
os.environ.setdefault("PYTHONEXECUTABLE", sys.executable)

_INSTANCE_KEY: Final[str] = "AUTOCAPTURE_ROOT_PID"
_EXPECTED_EXEC_KEY: Final[str] = "AUTOCAPTURE_EXPECTED_EXECUTABLE"


def _normalise(path: str) -> str:
    """Return a normalised absolute path for comparisons."""

    return str(Path(path).resolve())


def ensure_expected_interpreter(argv: Sequence[str] | None = None) -> bool:
    """Ensure the process runs under the preferred Python executable.

    When the orchestrator is launched as a Windows service, third-party modules
    have occasionally attempted to ``CreateProcess`` a new interpreter using the
    ``python`` found on ``PATH``. That interpreter may point at the system-wide
    installation instead of the virtual environment. By reserving the preferred
    interpreter path we can detect the mismatch and exit before any work starts.
    """

    expected = os.environ.get(_EXPECTED_EXEC_KEY)
    current = _normalise(sys.executable)

    if expected is None:
        os.environ[_EXPECTED_EXEC_KEY] = current
        return True

    if _normalise(expected) == current:
        return True

    if argv is None:
        argv = sys.argv[1:]

    sys.stderr.write(
        "Autocapture refused to start under unexpected interpreter.\n"
        f"Expected: {expected}\n"
        f"Current:  {sys.executable}\n"
    )

    # Returning ``False`` lets callers decide whether to exit or attempt a
    # manual re-exec using the expected interpreter.
    return False


def claim_single_instance() -> bool:
    """Return ``True`` if the current process is the primary orchestrator.

    Windows services occasionally spawn an auxiliary interpreter when third-party
    libraries rely on :mod:`multiprocessing` internals. Those auxiliary processes
    re-execute ``python -m autocapture.main`` which in turn would start a second
    capture loop. By reserving the instance token in an environment variable we
    can detect the secondary interpreter early and exit before any threads start.
    """

    current_pid = str(os.getpid())
    owner_pid = os.environ.setdefault(_INSTANCE_KEY, current_pid)
    return owner_pid == current_pid


__all__ = [
    "AppConfig",
    "load_config",
    "configure_logging",
    "claim_single_instance",
    "ensure_expected_interpreter",
]
