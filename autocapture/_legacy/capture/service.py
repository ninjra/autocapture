"""Event-driven capture orchestrator."""

from __future__ import annotations

import asyncio
import datetime as dt
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque
import uuid

from loguru import logger

from ..config import CaptureConfig
from ..logging_utils import get_logger
from .duplicate import DuplicateDetector
from .ffmpeg_recorder import SegmentRecorder
from .screen_capture import CaptureFrame, ScreenCaptureBackend


@dataclass(slots=True)
class CaptureEvent:
    frame: CaptureFrame
    output_path: Path


class CaptureService:
    """Manage HID triggers, GPU capture, dedupe, and dispatch to downstream queues."""

    def __init__(
        self,
        config: CaptureConfig,
        backend: ScreenCaptureBackend,
        on_capture: Callable[[CaptureEvent], None],
    ) -> None:
        self._config = config
        self._backend = backend
        self._on_capture = on_capture
        self._duplicate_detector = DuplicateDetector(config.hid.duplicate_threshold)
        self._loop = asyncio.get_event_loop()
        self._pending: Deque[CaptureEvent] = deque(maxlen=config.max_pending)
        self._running = threading.Event()
        self._log = get_logger("capture")
        self._staging_dir = config.staging_dir
        self._segment_recorder = SegmentRecorder(capture_config=config)
        self._last_activity: dt.datetime | None = None

    def start(self) -> None:
        self._running.set()
        self._backend.start()
        threading.Thread(target=self._run_capture_loop, daemon=True).start()
        logger.info("Capture service started")

    def stop(self) -> None:
        self._running.clear()
        self._backend.stop()
        self._segment_recorder.stop_segment()
        logger.info("Capture service stopped")

    def _run_capture_loop(self) -> None:  # pragma: no cover - requires event stream
        last_capture = dt.datetime.utcnow()
        while self._running.is_set():
            frame = self._backend.capture_once()
            now = dt.datetime.utcnow()
            self._maybe_stop_segment(now)
            if frame is None:
                continue
            delta_ms = (now - last_capture).total_seconds() * 1000
            if delta_ms < max(
                1000 / self._config.hid.fps_soft_cap, self._config.hid.min_interval_ms
            ):
                continue

            if frame.is_fullscreen and self._config.hid.block_fullscreen:
                self._log.debug(
                    "Skipping frame due to fullscreen focus: {}",
                    frame.foreground_process,
                )
                continue

            dup = self._duplicate_detector.update(frame.image)
            if dup.is_duplicate:
                self._log.debug(
                    "Dropping duplicate frame | window={} | distance={:.4f}",
                    frame.foreground_window,
                    dup.distance,
                )
                continue

            self._last_activity = now
            self._ensure_segment(now)
            self._segment_recorder.enqueue(frame)

            last_capture = now
            output_path = self._staging_dir / now.strftime("%Y/%m/%d/%H%M%S_%f.webp")
            frame.save(output_path, encoder="webp")
            event = CaptureEvent(frame=frame, output_path=output_path)
            if len(self._pending) >= self._pending.maxlen:  # type: ignore[arg-type]
                evicted = self._pending.popleft()
                self._log.warning(
                    "Dropping oldest capture due to backpressure: {}",
                    evicted.output_path,
                )
            self._pending.append(event)
            self._loop.call_soon_threadsafe(self._dispatch, event)

    def _dispatch(self, event: CaptureEvent) -> None:
        try:
            self._on_capture(event)
        except Exception as exc:  # pragma: no cover - logging side effect
            self._log.exception(
                "Failed to dispatch capture {}: {}", event.output_path, exc
            )

    def drain(self) -> list[CaptureEvent]:
        events = list(self._pending)
        self._pending.clear()
        return events

    def _ensure_segment(self, now: dt.datetime) -> None:
        if self._segment_recorder.start_segment(now, segment_id=str(uuid.uuid4())):
            return

    def _maybe_stop_segment(self, now: dt.datetime) -> None:
        if self._last_activity is None:
            return
        idle_ms = (now - self._last_activity).total_seconds() * 1000
        if idle_ms >= self._config.hid.idle_grace_ms:
            self._segment_recorder.stop_segment()
