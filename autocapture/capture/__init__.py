"""Capture subsystem entry points and helpers."""

from .service import CaptureService, CaptureEvent
from .duplicate import DuplicateDetector
from .screen_capture import ScreenCaptureBackend, CaptureFrame

__all__ = [
    "CaptureService",
    "CaptureEvent",
    "DuplicateDetector",
    "ScreenCaptureBackend",
    "CaptureFrame",
]
