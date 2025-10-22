"""Placeholder DirectX session wrapper.

This module should be replaced with a C++/Cython extension that interfaces with the
Windows Graphics Capture API. The stub exists to document the expected interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class RawFrame:
    image: Image.Image
    foreground_process: str
    foreground_window: str
    monitor_id: str
    is_fullscreen: bool


class DirectXSession:  # pragma: no cover - Windows implementation required
    def __init__(self, include_cursor: bool = True) -> None:
        self.include_cursor = include_cursor

    def initialize(self) -> None:
        raise NotImplementedError(
            "DirectXSession must be implemented with Windows Graphics Capture bindings."
        )

    def capture(self) -> Optional[RawFrame]:
        raise NotImplementedError

    def close(self) -> None:
        pass
