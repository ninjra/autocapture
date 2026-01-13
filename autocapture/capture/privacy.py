"""Privacy policy evaluation for capture decisions."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from ..config import PrivacyConfig
from ..logging_utils import get_logger
from ..tracking.types import ForegroundContext


@dataclass(frozen=True)
class PrivacyDecision:
    allowed: bool
    reason: str | None = None
    auto_pause: bool = False


class PrivacyPolicy:
    def __init__(self, privacy: PrivacyConfig) -> None:
        self._privacy = privacy
        self._log = get_logger("capture.privacy")
        self._processes = {name.lower() for name in privacy.exclude_processes}
        self._title_patterns = [
            re.compile(pattern)
            for pattern in privacy.exclude_window_title_regex
            if isinstance(pattern, str) and pattern
        ]

    def evaluate(
        self,
        context: ForegroundContext | None,
        *,
        screen_locked: bool,
        secure_desktop: bool,
    ) -> PrivacyDecision:
        if screen_locked:
            return PrivacyDecision(False, reason="screen_locked", auto_pause=True)
        if secure_desktop:
            return PrivacyDecision(False, reason="secure_desktop", auto_pause=True)
        if context is None:
            return PrivacyDecision(True)
        process_name = (context.process_name or "").lower()
        if process_name and process_name in self._processes:
            return PrivacyDecision(False, reason="process_denylist")
        title = context.window_title or ""
        for pattern in self._title_patterns:
            try:
                if pattern.search(title):
                    return PrivacyDecision(False, reason="title_denylist")
            except re.error as exc:
                self._log.warning("Invalid denylist regex %s: %s", pattern.pattern, exc)
        return PrivacyDecision(True)


def get_screen_lock_status() -> tuple[bool, bool]:
    """Return (screen_locked, secure_desktop) best-effort."""

    if sys.platform != "win32":
        return False, False
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return False, False

    user32 = ctypes.windll.user32
    DESKTOP_READOBJECTS = 0x0001
    DESKTOP_SWITCHDESKTOP = 0x0100
    UOI_NAME = 2

    hdesk = user32.OpenInputDesktop(0, False, DESKTOP_READOBJECTS | DESKTOP_SWITCHDESKTOP)
    if not hdesk:
        return True, True
    try:
        name_len = wintypes.DWORD(0)
        user32.GetUserObjectInformationW(hdesk, UOI_NAME, None, 0, ctypes.byref(name_len))
        if name_len.value == 0:
            return False, False
        buffer = ctypes.create_unicode_buffer(name_len.value)
        if not user32.GetUserObjectInformationW(
            hdesk, UOI_NAME, buffer, name_len, ctypes.byref(name_len)
        ):
            return False, False
        name = buffer.value.lower()
        if name in {"winlogon", "disconnect"}:
            return True, True
        if name != "default":
            return False, True
        return False, False
    finally:
        user32.CloseDesktop(hdesk)
