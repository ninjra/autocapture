from __future__ import annotations

from loguru import logger as loguru_logger

from autocapture.logging_utils import get_logger


def test_logger_adapter_formats_percent_and_braces() -> None:
    loguru_logger.remove()
    messages: list[str] = []
    loguru_logger.add(messages.append, format="{message}")

    log = get_logger("test")
    log.info("value {}", 123)
    log.info("value %s", 456)

    normalized = [message.strip() for message in messages]
    assert "value 123" in normalized
    assert "value 456" in normalized
