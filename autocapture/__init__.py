"""Autocapture package providing a high-performance local recall pipeline."""

from __future__ import annotations

import multiprocessing as _mp
import os
import sys

if hasattr(_mp, "set_executable"):
    _mp.set_executable(sys.executable)

# Hint for libraries which shell out to ``python`` (e.g. multiprocessing helpers)
# so they inherit the virtualenv interpreter instead of the system install.
os.environ.setdefault("PYTHONEXECUTABLE", sys.executable)

from .config import AppConfig, load_config
from .logging_utils import configure_logging

__all__ = ["AppConfig", "load_config", "configure_logging"]
