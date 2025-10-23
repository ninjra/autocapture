"""Autocapture package providing a high-performance local recall pipeline."""

from __future__ import annotations

import multiprocessing as _mp
import os
import sys
from typing import Final

from .config import AppConfig, load_config
from .logging_utils import configure_logging

if hasattr(_mp, "set_executable"):
    _mp.set_executable(sys.executable)

# Hint for libraries which shell out to ``python`` (e.g. multiprocessing helpers)
# so they inherit the virtualenv interpreter instead of the system install.
os.environ.setdefault("PYTHONEXECUTABLE", sys.executable)

_INSTANCE_KEY: Final[str] = "AUTOCAPTURE_ROOT_PID"


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


__all__ = ["AppConfig", "load_config", "configure_logging", "claim_single_instance"]
