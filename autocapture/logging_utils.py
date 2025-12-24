"""Structured logging configuration built on top of loguru."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def _default_log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "Autocapture" / "logs"
    return Path.home() / "AppData" / "Local" / "Autocapture" / "logs"


def configure_logging(log_dir: Path | str | None = None, level: str = "INFO") -> None:
    """Configure Loguru sinks for console and optional file output."""

    logger.remove()
    logger.configure(extra={"component": "app"})

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "pid={process} | thread={thread.name} | "
        "<cyan>{extra[component]}</cyan> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        format=log_format,
        colorize=True,
        level=level,
    )

    if log_dir is None:
        log_dir = _default_log_dir()

    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        logger.add(
            path / "autocapture.log",
            rotation="1 day",
            retention="14 days",
            compression="gz",
            level=level,
            backtrace=False,
            diagnose=False,
            format=log_format,
        )


def get_logger(name: Optional[str] = None):
    """Return a child logger with contextualized name."""

    if name:
        return logger.bind(component=name)
    return logger
