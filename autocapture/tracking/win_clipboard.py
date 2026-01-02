"""Windows clipboard metadata helpers (no content)."""

from __future__ import annotations

import sys

if sys.platform != "win32":

    def get_clipboard_sequence_number() -> int:
        return 0

    def clipboard_has_text() -> bool:
        return False

    def clipboard_has_image() -> bool:
        return False

else:
    import ctypes

    CF_UNICODETEXT = 13
    CF_DIB = 8

    def get_clipboard_sequence_number() -> int:
        return int(ctypes.windll.user32.GetClipboardSequenceNumber())

    def clipboard_has_text() -> bool:
        return bool(ctypes.windll.user32.IsClipboardFormatAvailable(CF_UNICODETEXT))

    def clipboard_has_image() -> bool:
        return bool(ctypes.windll.user32.IsClipboardFormatAvailable(CF_DIB))
