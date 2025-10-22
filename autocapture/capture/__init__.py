"""Capture subsystem entry points and helpers."""

from .duplicate import DuplicateDetector
from .screen_capture import CaptureFrame, DirectXDesktopDuplicator, ScreenCaptureBackend
from .service import CaptureEvent, CaptureService

__all__ = [
    "CaptureService",
    "CaptureEvent",
    "DuplicateDetector",
    "ScreenCaptureBackend",
    "CaptureFrame",
    "DirectXDesktopDuplicator",
]
