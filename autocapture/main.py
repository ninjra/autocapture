"""Application bootstrap."""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import os
import sys
from pathlib import Path

from loguru import logger

from . import claim_single_instance, ensure_expected_interpreter
from .capture import CaptureEvent, CaptureService, DirectXDesktopDuplicator
from .config import AppConfig
from .config import load_config
from .logging_utils import configure_logging
from .observability import MetricsServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autocapture orchestrator")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"))
    return parser.parse_args()
"""Legacy entry point shim.

Deprecated in favor of ``python -m autocapture``.
"""

from __future__ import annotations

from .__main__ import main


if __name__ == "__main__":
    main()
