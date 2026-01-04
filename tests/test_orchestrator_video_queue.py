from __future__ import annotations

from autocapture.capture import orchestrator as orch_module
from autocapture.capture.orchestrator import CaptureOrchestrator


def test_orchestrator_no_video_queue_property() -> None:
    assert not hasattr(CaptureOrchestrator, "video_queue")
    assert not hasattr(orch_module, "VideoBatch")
