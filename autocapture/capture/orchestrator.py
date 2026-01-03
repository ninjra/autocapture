"""Input-gated multi-monitor capture orchestration."""

from __future__ import annotations

import ctypes
import datetime as dt
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional
from uuid import uuid4

import numpy as np
from PIL import Image
from sqlalchemy import func, select

from ..logging_utils import get_logger
from ..storage.database import DatabaseManager
from ..storage.models import CaptureRecord, ObservationRecord, SegmentRecord
from ..tracking.win_foreground import get_foreground_context
from .backends import DxCamBackend, MssBackend, MonitorInfo
from .raw_input import RawInputListener


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


@dataclass(slots=True)
class ROIItem:
    image: np.ndarray
    captured_at: dt.datetime
    observation_id: str
    segment_id: Optional[str]
    cursor_x: int
    cursor_y: int
    monitor_id: str


@dataclass(slots=True)
class VideoBatch:
    captured_at: dt.datetime
    frames: Dict[str, np.ndarray]


class CaptureOrchestrator:
    """Coordinate raw input gating, multi-monitor capture, and ROI persistence."""

    def __init__(
        self,
        database: DatabaseManager,
        data_dir: Path,
        idle_grace_ms: int = 1500,
        fps_soft_cap: float = 4.0,
        roi_w: int = 512,
        roi_h: int = 512,
        roi_queue_size: int = 256,
        video_queue_size: int = 64,
        on_ocr_observation: Optional[Callable[[str], None]] = None,
        on_vision_observation: Optional[Callable[[str], None]] = None,
        vision_sample_rate: float = 0.0,
        on_segment_finalize: Optional[Callable[[str], None]] = None,
        hotkey_callback: Optional[Callable[[], None]] = None,
        raw_input: RawInputListener | None = None,
        ocr_backlog_soft_limit: int | None = None,
        ocr_backlog_check_s: float = 1.0,
    ) -> None:
        self._log = get_logger("orchestrator")
        self._database = database
        self._data_dir = data_dir
        self._idle_grace_ms = idle_grace_ms
        self._fps_soft_cap = fps_soft_cap
        self._roi_w = roi_w
        self._roi_h = roi_h
        self._on_ocr_observation = on_ocr_observation
        self._on_vision_observation = on_vision_observation
        self._vision_sample_rate = vision_sample_rate
        self._on_segment_finalize = on_segment_finalize
        self._backend = self._select_backend()
        self._monitors = self._backend.monitors
        self._raw_input = raw_input or RawInputListener(
            idle_grace_ms=idle_grace_ms,
            on_activity=None,
            on_hotkey=hotkey_callback,
        )
        self._roi_queue: queue.Queue[ROIItem] = queue.Queue(maxsize=roi_queue_size)
        self._video_queue: queue.Queue[VideoBatch] = queue.Queue(maxsize=video_queue_size)
        self._running = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._roi_thread: Optional[threading.Thread] = None
        self._active_segment_id: Optional[str] = None
        self._was_active = False
        self._ocr_backlog_soft_limit = ocr_backlog_soft_limit
        self._ocr_backlog_check_s = ocr_backlog_check_s
        self._ocr_backlog_last_check = 0.0
        self._ocr_backlog_cached_throttle = False
        self._ocr_backlog_last_count = 0
        self._ocr_backlog_last_log = 0.0

    @property
    def video_queue(self) -> queue.Queue[VideoBatch]:
        return self._video_queue

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._raw_input.start()
        self._roi_thread = threading.Thread(target=self._run_roi_saver, daemon=True)
        self._roi_thread.start()
        self._capture_thread = threading.Thread(
            target=self._run_capture_loop, daemon=True
        )
        self._capture_thread.start()
        self._log.info("Capture orchestrator started")

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        self._raw_input.stop()
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._roi_thread:
            self._roi_thread.join(timeout=2.0)
        self._log.info("Capture orchestrator stopped")

    def _run_capture_loop(self) -> None:  # pragma: no cover - depends on Windows APIs
        interval = 1.0 / max(self._fps_soft_cap, 0.01)
        while self._running.is_set():
            loop_start = time.monotonic()
            now_ms = int(loop_start * 1000)
            active = now_ms < self._raw_input.active_until_ts

            if active:
                if not self._was_active:
                    self._active_segment_id = self._start_segment()
                self._capture_tick()
                elapsed = time.monotonic() - loop_start
                sleep_for = max(0.0, interval - elapsed)
                time.sleep(sleep_for)
            else:
                if self._was_active:
                    self._close_segment(self._active_segment_id)
                    self._active_segment_id = None
                time.sleep(0.1)

            self._was_active = active

    def _capture_tick(self) -> None:
        frames = self._backend.grab_all()
        if not frames:
            return
        cursor_x, cursor_y = self._get_cursor_pos()
        monitor = self._find_monitor(cursor_x, cursor_y)
        if monitor and monitor.id in frames:
            roi = self._crop_roi(frames[monitor.id], monitor, cursor_x, cursor_y)
            observation_id = str(uuid4())
            roi_item = ROIItem(
                image=roi,
                captured_at=dt.datetime.now(dt.timezone.utc),
                observation_id=observation_id,
                segment_id=self._active_segment_id,
                cursor_x=cursor_x,
                cursor_y=cursor_y,
                monitor_id=monitor.id,
            )
            self._enqueue_roi(roi_item)
        self._enqueue_video(VideoBatch(captured_at=dt.datetime.now(dt.timezone.utc), frames=frames))

    def _enqueue_roi(self, item: ROIItem) -> None:
        if self._should_throttle_ocr():
            return
        try:
            self._roi_queue.put_nowait(item)
        except queue.Full:
            self._log.warning("ROI queue full; dropping observation %s", item.observation_id)

    def _enqueue_video(self, batch: VideoBatch) -> None:
        try:
            self._video_queue.put_nowait(batch)
        except queue.Full:
            self._log.warning("Video queue full; dropping batch at %s", batch.captured_at)

    def _run_roi_saver(self) -> None:
        while self._running.is_set() or not self._roi_queue.empty():
            try:
                item = self._roi_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                path = self._save_roi(item)
                self._persist_observation(item, path)
                if self._on_ocr_observation:
                    self._on_ocr_observation(item.observation_id)
                if (
                    self._on_vision_observation
                    and self._vision_sample_rate > 0
                    and random.random() < self._vision_sample_rate
                ):
                    self._on_vision_observation(item.observation_id)
            except Exception as exc:  # pragma: no cover - side effects
                self._log.exception(
                    "Failed to persist ROI observation %s: %s",
                    item.observation_id,
                    exc,
                )
            finally:
                self._roi_queue.task_done()

    def _save_roi(self, item: ROIItem) -> Path:
        timestamp = item.captured_at.astimezone(dt.timezone.utc)
        path = (
            self._data_dir
            / "media"
            / "roi"
            / timestamp.strftime("%Y/%m/%d")
            / f"{timestamp.strftime('%H%M%S_%f')}_{item.observation_id}.webp"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = item.image[:, :, ::-1]
        image = Image.fromarray(np.ascontiguousarray(rgb))
        image.save(path, format="WEBP", quality=80, method=6)
        return path

    def _persist_observation(self, item: ROIItem, path: Path) -> None:
        ctx = get_foreground_context()
        foreground_process = ctx.process_name if ctx else "unknown"
        foreground_window = ctx.window_title if ctx else "unknown"
        with self._database.session() as session:
            session.add(
                ObservationRecord(
                    id=item.observation_id,
                    captured_at=item.captured_at,
                    image_path=str(path),
                    segment_id=item.segment_id,
                    cursor_x=item.cursor_x,
                    cursor_y=item.cursor_y,
                    monitor_id=item.monitor_id,
                )
            )
            session.add(
                CaptureRecord(
                    id=item.observation_id,
                    captured_at=item.captured_at,
                    image_path=str(path),
                    foreground_process=foreground_process or "unknown",
                    foreground_window=foreground_window or "unknown",
                    monitor_id=item.monitor_id,
                    is_fullscreen=False,
                    ocr_status="pending",
                )
            )

    def _start_segment(self) -> str:
        segment_id = str(uuid4())
        with self._database.session() as session:
            session.add(
                SegmentRecord(
                    id=segment_id,
                    started_at=dt.datetime.now(dt.timezone.utc),
                    state="recording",
                )
            )
        return segment_id

    def _close_segment(self, segment_id: Optional[str]) -> None:
        if not segment_id:
            return
        with self._database.session() as session:
            segment = session.get(SegmentRecord, segment_id)
            if segment:
                segment.ended_at = dt.datetime.now(dt.timezone.utc)
                segment.state = "closed"
        if self._on_segment_finalize:
            self._on_segment_finalize(segment_id)

    def _select_backend(self):
        try:
            backend = DxCamBackend()
            test = backend.grab_all()
            if test:
                return backend
            self._log.warning("DxCam produced empty capture; falling back to MSS")
        except Exception as exc:  # pragma: no cover - depends on dxcam
            self._log.warning("DxCam unavailable; falling back to MSS: %s", exc)
        return MssBackend()

    def _find_monitor(self, x: int, y: int) -> Optional[MonitorInfo]:
        for monitor in self._monitors:
            if monitor.contains(x, y):
                return monitor
        return self._monitors[0] if self._monitors else None

    def _crop_roi(
        self, frame: np.ndarray, monitor: MonitorInfo, x: int, y: int
    ) -> np.ndarray:
        local_x = x - monitor.left
        local_y = y - monitor.top
        half_w = self._roi_w // 2
        half_h = self._roi_h // 2
        x0 = max(0, local_x - half_w)
        y0 = max(0, local_y - half_h)
        x1 = min(frame.shape[1], x0 + self._roi_w)
        y1 = min(frame.shape[0], y0 + self._roi_h)
        x0 = max(0, x1 - self._roi_w)
        y0 = max(0, y1 - self._roi_h)
        return frame[y0:y1, x0:x1]

    @staticmethod
    def _get_cursor_pos() -> tuple[int, int]:
        point = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
        return int(point.x), int(point.y)

    def _should_throttle_ocr(self) -> bool:
        if not self._ocr_backlog_soft_limit:
            return False
        now = time.monotonic()
        if now - self._ocr_backlog_last_check < self._ocr_backlog_check_s:
            return self._ocr_backlog_cached_throttle
        self._ocr_backlog_last_check = now
        try:
            with self._database.session() as session:
                count = session.execute(
                    select(func.count())
                    .select_from(CaptureRecord)
                    .where(CaptureRecord.ocr_status.in_(["pending", "processing"]))
                ).scalar_one()
        except Exception as exc:  # pragma: no cover - defensive
            self._log.debug("Failed to check OCR backlog: %s", exc)
            self._ocr_backlog_cached_throttle = False
            return False
        self._ocr_backlog_last_count = int(count or 0)
        self._ocr_backlog_cached_throttle = (
            self._ocr_backlog_last_count >= self._ocr_backlog_soft_limit
        )
        if self._ocr_backlog_cached_throttle and now - self._ocr_backlog_last_log > 5.0:
            self._ocr_backlog_last_log = now
            self._log.warning(
                "OCR backlog at %s (limit %s); throttling capture",
                self._ocr_backlog_last_count,
                self._ocr_backlog_soft_limit,
            )
        return self._ocr_backlog_cached_throttle
