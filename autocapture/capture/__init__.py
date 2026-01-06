"""Capture subsystem entry points and helpers."""

from .duplicate import DuplicateDetector
from .screen_capture import CaptureFrame, DirectXDesktopDuplicator, ScreenCaptureBackend

__all__ = [
    "DuplicateDetector",
    "ScreenCaptureBackend",
    "CaptureFrame",
    "DirectXDesktopDuplicator",
]
