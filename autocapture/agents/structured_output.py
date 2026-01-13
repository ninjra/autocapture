"""Structured JSON parsing helpers for agent outputs."""

from __future__ import annotations

import json
import re
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)


class StructuredOutputError(RuntimeError):
    pass


def extract_json_payload(text: str) -> str:
    if not text:
        raise StructuredOutputError("Empty model output.")
    fenced = _FENCE_RE.findall(text)
    if fenced:
        text = "\n".join(fenced).strip()
    match = _JSON_RE.search(text)
    if not match:
        raise StructuredOutputError("No JSON object found in output.")
    return match.group(0).strip()


def parse_structured_output(
    raw_text: str,
    model: type[T],
    *,
    repair_fn: Callable[[str], str] | None = None,
) -> T:
    try:
        payload = extract_json_payload(raw_text)
        data = json.loads(payload)
        return model.model_validate(data)
    except (StructuredOutputError, json.JSONDecodeError, ValidationError) as exc:
        if not repair_fn:
            raise StructuredOutputError(f"Failed to parse structured output: {exc}") from exc
        repaired = repair_fn(raw_text)
        try:
            payload = extract_json_payload(repaired)
            data = json.loads(payload)
            return model.model_validate(data)
        except (StructuredOutputError, json.JSONDecodeError, ValidationError) as exc2:
            raise StructuredOutputError(f"Repair attempt failed: {exc2}") from exc2
