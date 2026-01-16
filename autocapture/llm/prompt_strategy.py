"""Prompt strategy engine for repetition and step-by-step prompting."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from ..logging_utils import get_logger

_LOG = get_logger("llm.prompt_strategy")
_TEXT_SEGMENT_TYPES = {"text", "input_text", "output_text"}


class PromptStrategy(str, Enum):
    BASELINE = "baseline"
    REPEAT_2X = "repeat_2x"
    REPEAT_3X = "repeat_3x"
    STEP_BY_STEP = "step_by_step"
    STEP_BY_STEP_PLUS_REPEAT_2X = "step_by_step_plus_repeat_2x"


@dataclass(frozen=True)
class PromptStrategySettings:
    strategy_default: PromptStrategy
    prompt_repeat_factor: int
    enable_step_by_step: bool
    step_by_step_phrase: str
    step_by_step_two_stage: bool
    max_prompt_chars_for_repetition: int
    max_tokens_headroom: int
    max_context_tokens: int | None
    force_no_reasoning: bool
    strategy_auto_mode: bool
    repetition_delimiter: str
    store_prompt_transforms: bool
    prompt_store_redaction: bool
    data_dir: Path | None

    @classmethod
    def from_llm_config(
        cls, llm_config: Any, *, data_dir: Path | None = None
    ) -> "PromptStrategySettings":
        return cls(
            strategy_default=_parse_strategy(llm_config.prompt_strategy_default),
            prompt_repeat_factor=llm_config.prompt_repeat_factor,
            enable_step_by_step=llm_config.enable_step_by_step,
            step_by_step_phrase=llm_config.step_by_step_phrase,
            step_by_step_two_stage=llm_config.step_by_step_two_stage,
            max_prompt_chars_for_repetition=llm_config.max_prompt_chars_for_repetition,
            max_tokens_headroom=llm_config.max_tokens_headroom,
            max_context_tokens=llm_config.max_context_tokens,
            force_no_reasoning=llm_config.force_no_reasoning,
            strategy_auto_mode=llm_config.strategy_auto_mode,
            repetition_delimiter=llm_config.prompt_repetition_delimiter,
            store_prompt_transforms=llm_config.store_prompt_transforms,
            prompt_store_redaction=llm_config.prompt_store_redaction,
            data_dir=data_dir,
        )


@dataclass(frozen=True)
class PromptStrategyMetadata:
    strategy: PromptStrategy
    repeat_factor: int
    step_by_step_used: bool
    prompt_chars: int
    prompt_tokens_estimate: int
    prompt_hash_before: str
    prompt_hash_after: str
    degraded_from: PromptStrategy | None
    degraded_reason: str | None
    safe_mode_degraded: bool
    task_type: str | None
    max_context_tokens: int | None
    max_tokens_headroom: int
    repetition_delimiter: str
    applied_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "repeat_factor": self.repeat_factor,
            "step_by_step_used": self.step_by_step_used,
            "prompt_chars": self.prompt_chars,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "prompt_hash_before": self.prompt_hash_before,
            "prompt_hash_after": self.prompt_hash_after,
            "degraded_from": self.degraded_from.value if self.degraded_from else None,
            "degraded_reason": self.degraded_reason,
            "safe_mode_degraded": self.safe_mode_degraded,
            "task_type": self.task_type,
            "max_context_tokens": self.max_context_tokens,
            "max_tokens_headroom": self.max_tokens_headroom,
            "repetition_delimiter": self.repetition_delimiter,
            "applied_at": self.applied_at,
        }


@dataclass(frozen=True)
class PromptStrategyResult:
    messages: list[dict[str, Any]]
    metadata: PromptStrategyMetadata


def apply_prompt_strategy(
    messages: Sequence[dict[str, Any]],
    settings: PromptStrategySettings,
    *,
    task_type: str | None = None,
    step_by_step_requested: bool | None = None,
    override_strategy: PromptStrategy | None = None,
) -> PromptStrategyResult:
    message_list = [copy.deepcopy(message) for message in messages]
    step_by_step_requested = (
        settings.enable_step_by_step if step_by_step_requested is None else step_by_step_requested
    )
    if settings.force_no_reasoning and step_by_step_requested:
        _LOG.info(
            json.dumps(
                {
                    "event": "prompt_strategy.step_by_step_disabled",
                    "reason": "force_no_reasoning",
                    "task_type": task_type,
                }
            )
        )
        step_by_step_requested = False

    strategy = override_strategy or _select_strategy(settings, step_by_step_requested)
    degraded_from: PromptStrategy | None = None
    degraded_reason: str | None = None
    safe_mode_degraded = False

    message_list = _apply_step_by_step_phrase(message_list, settings, step_by_step_requested)
    pre_text = _serialize_messages(message_list)
    prompt_chars = len(pre_text)
    pre_hash = _hash_text(pre_text)

    repeat_factor = _repeat_factor_for_strategy(strategy, settings.prompt_repeat_factor)
    if repeat_factor > 1:
        strategy, degraded_from, degraded_reason, safe_mode_degraded = _degrade_if_needed(
            strategy,
            repeat_factor,
            prompt_chars,
            settings,
        )
        repeat_factor = _repeat_factor_for_strategy(strategy, settings.prompt_repeat_factor)
        if repeat_factor > 1:
            message_list = _apply_repetition(
                message_list, settings.repetition_delimiter, repeat_factor
            )
        else:
            repeat_factor = 1
    prompt_tokens_estimate = _estimate_tokens(_serialize_messages(message_list))
    post_text = _serialize_messages(message_list)
    post_hash = _hash_text(post_text)

    step_by_step_used = step_by_step_requested and settings.enable_step_by_step
    effective_strategy = strategy
    if step_by_step_used and repeat_factor > 1 and strategy == PromptStrategy.REPEAT_2X:
        effective_strategy = PromptStrategy.STEP_BY_STEP_PLUS_REPEAT_2X
    metadata = PromptStrategyMetadata(
        strategy=effective_strategy,
        repeat_factor=repeat_factor,
        step_by_step_used=step_by_step_used,
        prompt_chars=len(post_text),
        prompt_tokens_estimate=prompt_tokens_estimate,
        prompt_hash_before=pre_hash,
        prompt_hash_after=post_hash,
        degraded_from=degraded_from,
        degraded_reason=degraded_reason,
        safe_mode_degraded=safe_mode_degraded,
        task_type=task_type,
        max_context_tokens=settings.max_context_tokens,
        max_tokens_headroom=settings.max_tokens_headroom,
        repetition_delimiter=settings.repetition_delimiter,
        applied_at=time.time(),
    )
    _LOG.info(
        json.dumps(
            {
                "event": "prompt_strategy.applied",
                **metadata.to_dict(),
            }
        )
    )
    _record_prompt_transform(settings, metadata, pre_text, post_text)
    return PromptStrategyResult(messages=message_list, metadata=metadata)


def _parse_strategy(value: str) -> PromptStrategy:
    normalized = value.strip().lower()
    for strategy in PromptStrategy:
        if strategy.value == normalized:
            return strategy
    return PromptStrategy.BASELINE


def _select_strategy(settings: PromptStrategySettings, step_by_step: bool) -> PromptStrategy:
    if settings.strategy_auto_mode:
        if step_by_step and settings.enable_step_by_step:
            return PromptStrategy.STEP_BY_STEP
        return (
            PromptStrategy.REPEAT_3X
            if settings.prompt_repeat_factor == 3
            else PromptStrategy.REPEAT_2X
        )
    return settings.strategy_default


def _repeat_factor_for_strategy(strategy: PromptStrategy, default_factor: int) -> int:
    if strategy in {PromptStrategy.REPEAT_2X, PromptStrategy.STEP_BY_STEP_PLUS_REPEAT_2X}:
        return 2
    if strategy == PromptStrategy.REPEAT_3X:
        return 3
    if strategy == PromptStrategy.BASELINE or strategy == PromptStrategy.STEP_BY_STEP:
        return 1
    return max(1, default_factor)


def _degrade_if_needed(
    strategy: PromptStrategy,
    repeat_factor: int,
    prompt_chars: int,
    settings: PromptStrategySettings,
) -> tuple[PromptStrategy, PromptStrategy | None, str | None, bool]:
    degraded_from: PromptStrategy | None = None
    degraded_reason: str | None = None
    safe_mode_degraded = False
    if repeat_factor <= 1:
        return strategy, degraded_from, degraded_reason, safe_mode_degraded

    def _too_large(repeat_factor_inner: int) -> bool:
        repeated_chars = prompt_chars + (repeat_factor_inner - 1) * (
            prompt_chars + len(settings.repetition_delimiter)
        )
        if prompt_chars > settings.max_prompt_chars_for_repetition:
            return True
        if repeated_chars > settings.max_prompt_chars_for_repetition:
            return True
        if settings.max_context_tokens:
            est_tokens = max(1, repeated_chars // 4)
            return est_tokens + settings.max_tokens_headroom > settings.max_context_tokens
        return False

    while repeat_factor > 1 and _too_large(repeat_factor):
        degraded_from = strategy if degraded_from is None else degraded_from
        safe_mode_degraded = True
        degraded_reason = "context_limit"
        strategy = _step_down_strategy(strategy)
        repeat_factor = _repeat_factor_for_strategy(strategy, settings.prompt_repeat_factor)
        if repeat_factor <= 1:
            break

    if safe_mode_degraded:
        _LOG.warning(
            json.dumps(
                {
                    "event": "prompt_strategy.degraded",
                    "from": degraded_from.value if degraded_from else None,
                    "to": strategy.value,
                    "reason": degraded_reason,
                    "prompt_chars": prompt_chars,
                    "repeat_factor": repeat_factor,
                }
            )
        )
    return strategy, degraded_from, degraded_reason, safe_mode_degraded


def _step_down_strategy(strategy: PromptStrategy) -> PromptStrategy:
    if strategy == PromptStrategy.REPEAT_3X:
        return PromptStrategy.REPEAT_2X
    if strategy == PromptStrategy.REPEAT_2X:
        return PromptStrategy.BASELINE
    if strategy == PromptStrategy.STEP_BY_STEP_PLUS_REPEAT_2X:
        return PromptStrategy.STEP_BY_STEP
    return PromptStrategy.BASELINE


def _apply_step_by_step_phrase(
    messages: list[dict[str, Any]],
    settings: PromptStrategySettings,
    step_by_step_requested: bool,
) -> list[dict[str, Any]]:
    if not step_by_step_requested or not settings.enable_step_by_step:
        return messages
    idx = _find_last_user_index(messages)
    if idx is None:
        return messages
    messages[idx] = _append_text(
        messages[idx],
        f"\n\n{settings.step_by_step_phrase.strip()}",
    )
    return messages


def _apply_repetition(
    messages: list[dict[str, Any]],
    delimiter: str,
    repeat_factor: int,
) -> list[dict[str, Any]]:
    if repeat_factor <= 1:
        return messages
    idx = _find_last_user_index(messages)
    if idx is None:
        return messages
    repeated = messages[idx]
    text = _message_text(repeated)
    if not text:
        return messages
    repeat_text = (delimiter + text) * (repeat_factor - 1)
    messages[idx] = _append_text(repeated, repeat_text)
    return messages


def _append_text(message: dict[str, Any], text: str) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, list):
        segment_type = _text_segment_type(content) or "text"
        content.append({"type": segment_type, "text": text})
        message["content"] = content
        return message
    if isinstance(content, str):
        message["content"] = content + text
        return message
    message["content"] = text
    return message


def _text_segment_type(content: list[Any]) -> str | None:
    for segment in content:
        if isinstance(segment, dict) and segment.get("type") in _TEXT_SEGMENT_TYPES:
            return segment.get("type")
    return None


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_segment_text(segment) for segment in content)
    return ""


def _segment_text(segment: Any) -> str:
    if not isinstance(segment, dict):
        return ""
    if segment.get("type") in _TEXT_SEGMENT_TYPES and "text" in segment:
        return str(segment["text"])
    if "text" in segment and segment.get("type") is None:
        return str(segment["text"])
    return ""


def _find_last_user_index(messages: Sequence[dict[str, Any]]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            return idx
    return None


def _serialize_messages(messages: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        lines.append(f"[{role}]")
        lines.append(_message_text(message))
    return "\n".join(lines)


def render_prompt_text(messages: Sequence[dict[str, Any]]) -> str:
    return _serialize_messages(messages)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_prompt_transform(
    settings: PromptStrategySettings,
    metadata: PromptStrategyMetadata,
    pre_text: str,
    post_text: str,
) -> None:
    payload = {
        "metadata": metadata.to_dict(),
    }
    if not settings.store_prompt_transforms:
        return
    if settings.data_dir is None:
        return
    store_dir = settings.data_dir / "promptops" / "prompt_transforms"
    store_dir.mkdir(parents=True, exist_ok=True)
    record_id = f"{metadata.prompt_hash_after[:12]}_{int(metadata.applied_at)}"
    record_path = store_dir / f"{record_id}.json"
    if settings.prompt_store_redaction:
        payload["prompt_before"] = _redact_prompt(pre_text)
        payload["prompt_after"] = _redact_prompt(post_text)
    else:
        payload["prompt_before"] = pre_text
        payload["prompt_after"] = post_text
    record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_REDACT_TOKEN = re.compile(r"[A-Za-z0-9_\-]{24,}")


def _redact_prompt(text: str) -> str:
    return _REDACT_TOKEN.sub("[REDACTED]", text)
