"""Datatypes for host vector tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class ForegroundContext:
    process_name: str
    window_title: str
    pid: int | None = None
    hwnd: int | None = None


@dataclass(slots=True)
class InputVectorEvent:
    ts_ms: int
    device: Literal["keyboard", "mouse", "hid"]
    mouse: dict[str, int] | None = None


@dataclass(slots=True)
class ForegroundChangeEvent:
    ts_ms: int
    new: ForegroundContext
    old: ForegroundContext | None


@dataclass(slots=True)
class ClipboardChangeEvent:
    ts_ms: int
    sequence: int
    has_text: bool
    has_image: bool


@dataclass(slots=True)
class HostEventRow:
    id: str
    ts_start_ms: int
    ts_end_ms: int
    kind: str
    session_id: str | None
    app_name: str | None
    window_title: str | None
    payload_json: str
