"""Prompt repetition helpers for LLM payloads."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from ..logging_utils import get_logger

_LOG = get_logger("llm.prompt_repetition")
_TEXT_SEGMENT_TYPES = {"text", "input_text"}


def apply_prompt_repetition(
    messages: Sequence[dict[str, Any]],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    """Return a prompt list with the body repeated once when enabled."""

    if not enabled:
        return messages if isinstance(messages, list) else list(messages)

    message_list = list(messages)
    system_count = 0
    for message in message_list:
        if message.get("role") == "system":
            system_count += 1
        else:
            break

    body_messages = message_list[system_count:]
    repeated_messages: list[dict[str, Any]] = []
    image_message_count = 0
    stripped_count = 0
    for message in body_messages:
        if _message_has_images(message):
            image_message_count += 1
        sanitized, stripped = _strip_images(copy.deepcopy(message))
        if sanitized is None:
            continue
        if stripped:
            stripped_count += 1
        repeated_messages.append(sanitized)

    _LOG.debug(
        "Prompt repetition enabled=%s system_messages=%s body_messages=%s repeated_messages=%s "
        "image_messages=%s stripped_messages=%s",
        enabled,
        system_count,
        len(body_messages),
        len(repeated_messages),
        image_message_count,
        stripped_count,
    )

    original_messages = [copy.deepcopy(message) for message in message_list]
    return original_messages + repeated_messages


def _strip_images(message: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    stripped = False
    if "images" in message:
        message.pop("images", None)
        stripped = True
    content = message.get("content")
    if isinstance(content, list):
        text_segments = [segment for segment in content if _is_text_segment(segment)]
        if len(text_segments) != len(content):
            stripped = True
        if not text_segments:
            return None, stripped
        message["content"] = text_segments
    return message, stripped


def _is_text_segment(segment: Any) -> bool:
    if not isinstance(segment, dict):
        return False
    segment_type = segment.get("type")
    if segment_type in _TEXT_SEGMENT_TYPES:
        return True
    if segment_type is None and "text" in segment:
        return True
    return False


def _message_has_images(message: dict[str, Any]) -> bool:
    if message.get("images"):
        return True
    content = message.get("content")
    if isinstance(content, list):
        return any(not _is_text_segment(segment) for segment in content)
    return False
