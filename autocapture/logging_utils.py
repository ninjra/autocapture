"""Structured logging configuration built on top of loguru."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def configure_logging(log_dir: Path | str | None = None, level: str = "INFO") -> None:
    """Configure Loguru sinks for console and optional file output."""

    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        level=level,
    )

    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        logger.add(
            path / "autocapture.log",
            rotation="1 day",
            retention="14 days",
            compression="gz",
            level=level,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )


def get_logger(name: Optional[str] = None):
    """Return a child logger with contextualized name."""

    if name:
        return logger.bind(component=name)
    return logger
