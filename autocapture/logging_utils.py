"""Structured logging configuration built on top of loguru."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from .security.redaction import redact_mapping, redact_text


class LoggerAdapter:
    def __init__(self, base_logger):
        self._logger = base_logger

    def bind(self, **kwargs):
        return LoggerAdapter(self._logger.bind(**kwargs))

    def debug(self, message, *args, **kwargs):
        return self._log("DEBUG", message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        return self._log("INFO", message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        return self._log("WARNING", message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        return self._log("ERROR", message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        return self._log("CRITICAL", message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        message, args = _prepare_message(message, args)
        return self._logger.exception(message, *args, **kwargs)

    def _log(self, level: str, message, *args, **kwargs):
        message, args = _prepare_message(message, args)
        return self._logger.log(level, message, *args, **kwargs)


def _prepare_message(message, args):
    redacted_args = tuple(redact_text(arg) if isinstance(arg, str) else arg for arg in (args or ()))
    if args and isinstance(message, str):
        if "%" in message:
            try:
                message = message % redacted_args
                return redact_text(message), ()
            except Exception:
                return redact_text(message), redacted_args
        if "{" in message:
            try:
                message = message.format(*redacted_args)
                return redact_text(message), ()
            except Exception:
                return redact_text(message), redacted_args
    if isinstance(message, str):
        return redact_text(message), redacted_args
    return message, redacted_args


def _default_log_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "Autocapture" / "logs"
        return Path.home() / "AppData" / "Local" / "Autocapture" / "logs"

    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home) / "autocapture" / "logs"
    return Path.home() / ".local" / "state" / "autocapture" / "logs"


def configure_logging(log_dir: Path | str | None = None, level: str = "INFO") -> None:
    """Configure Loguru sinks for console and optional file output."""

    logger.remove()
    logger.configure(
        extra={"component": "app"},
        patcher=lambda record: _redact_record(record),
    )

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
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to create log directory {} ({}). File logging disabled.",
                path,
                exc,
            )
            return
        try:
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
        except OSError as exc:
            logger.warning(
                "Failed to initialize file logging {} ({}). File logging disabled.",
                path,
                exc,
            )


def _redact_record(record: dict) -> None:
    if "message" in record and isinstance(record["message"], str):
        record["message"] = redact_text(record["message"])
    extra = record.get("extra")
    if isinstance(extra, dict):
        record["extra"] = redact_mapping(extra)


def get_logger(name: Optional[str] = None):
    """Return a child logger with contextualized name."""

    if name:
        return LoggerAdapter(logger.bind(component=name))
    return LoggerAdapter(logger)
