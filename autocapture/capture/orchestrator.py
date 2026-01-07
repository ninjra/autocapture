"""Input-gated multi-monitor capture orchestration."""

from __future__ import annotations

import ctypes
import datetime as dt
import queue
import random
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional
from uuid import uuid4

import numpy as np
from PIL import Image
from sqlalchemy import func, select

from ..config import CaptureConfig, WorkerConfig
from ..logging_utils import get_logger
from ..observability.metrics import (
    captures_skipped_backpressure_total,
    captures_taken_total,
    disk_low_total,
    ocr_backlog,
    ocr_backlog_gauge,
    ocr_stale_processing_gauge,
    roi_queue_depth,
    roi_queue_full_total,
    video_backpressure_events_total,
    video_disabled,
)
from ..media.store import MediaStore
from ..storage.database import DatabaseManager
from ..storage.models import CaptureRecord, ObservationRecord, SegmentRecord
from ..tracking.win_foreground import get_foreground_context, is_fullscreen_window
from .backends import DxCamBackend, MssBackend, MonitorInfo
from .duplicate import DuplicateDetector
from .ffmpeg_recorder import SegmentRecorder
from .raw_input import RawInputListener
from .screen_capture import CaptureFrame


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
    foreground_process: str
    foreground_window: str
    is_fullscreen: bool


class CaptureOrchestrator:
    """Coordinate raw input gating, multi-monitor capture, and ROI persistence."""

    def __init__(
        self,
        database: DatabaseManager,
        capture_config: CaptureConfig,
        worker_config: WorkerConfig,
        roi_w: int = 512,
        roi_h: int = 512,
        roi_queue_size: int = 256,
        on_ocr_observation: Optional[Callable[[str], None]] = None,
        on_vision_observation: Optional[Callable[[str], None]] = None,
        vision_sample_rate: float = 0.0,
        on_segment_finalize: Optional[Callable[[str], None]] = None,
        hotkey_callback: Optional[Callable[[], None]] = None,
        raw_input: RawInputListener | None = None,
        ocr_backlog_check_s: float = 1.0,
        media_store: MediaStore | None = None,
        backend: object | None = None,
    ) -> None:
        self._log = get_logger("orchestrator")
        self._database = database
        self._capture_config = capture_config
        self._worker_config = worker_config
        self._roi_w = roi_w
        self._roi_h = roi_h
        self._on_ocr_observation = on_ocr_observation
        self._on_vision_observation = on_vision_observation
        self._vision_sample_rate = vision_sample_rate
        self._on_segment_finalize = on_segment_finalize
        self._backend = backend or self._select_backend()
        self._monitors = self._backend.monitors
        self._raw_input = raw_input or RawInputListener(
            idle_grace_ms=capture_config.hid.idle_grace_ms,
            on_activity=None,
            on_hotkey=hotkey_callback,
        )
        self._roi_queue: queue.Queue[ROIItem] = queue.Queue(maxsize=roi_queue_size)
        self._segment_recorder = SegmentRecorder(capture_config=capture_config)
        self._duplicate_detector = DuplicateDetector(
            threshold=capture_config.hid.duplicate_threshold,
            window_s=capture_config.hid.duplicate_window_s,
            max_items=capture_config.hid.duplicate_max_items,
            pixel_threshold=capture_config.hid.duplicate_pixel_threshold,
        )
        self._media_store = media_store
        self._segment_video_paths: dict[str, Path] = {}
        self._running = threading.Event()
        self._paused = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._roi_thread: Optional[threading.Thread] = None
        self._active_segment_id: Optional[str] = None
        self._was_active = False
        self._ocr_backlog_soft_limit = worker_config.ocr_backlog_soft_limit
        self._ocr_backlog_check_s = ocr_backlog_check_s
        self._ocr_backlog_last_check = 0.0
        self._ocr_backlog_cached_throttle = False
        self._ocr_backlog_last_count = 0
        self._ocr_backlog_last_log = 0.0
        self._backoff_s = 0.0
        self._video_sampling_divisor = 1
        self._video_tick = 0
        self._video_drop_window: deque[float] = deque(maxlen=200)
        self._video_drop_window_s = 5.0
        self._video_disabled_until = 0.0
        self._video_last_drop = 0.0
        self._state_lock = threading.Lock()

    def start(self) -> None:
        with self._state_lock:
            if self._running.is_set():
                return
            self._running.set()
            self._paused.clear()

        self._raw_input.start()
        self._roi_thread = threading.Thread(
            target=self._run_roi_saver, daemon=True, name="autocapture-roi-saver"
        )
        self._roi_thread.start()
        self._capture_thread = threading.Thread(
            target=self._run_capture_loop,
            daemon=True,
            name="autocapture-capture-loop",
        )
        self._capture_thread.start()
        self._log.info("Capture orchestrator started")

    def stop(self) -> None:
        with self._state_lock:
            if not self._running.is_set():
                return
            self._running.clear()
            self._paused.clear()

        self._raw_input.stop()

        if self._capture_thread:
            self._capture_thread.join(timeout=5.0)

        if self._capture_thread and self._capture_thread.is_alive():
            self._log.warning("Capture thread did not stop in time; forcing shutdown.")
            segment_id = self._active_segment_id
            self._active_segment_id = None
            try:
                self._close_segment(segment_id)
            except Exception as exc:
                self._log.exception("Forced segment close failed: {}", exc)
                try:
                    self._segment_recorder.stop_segment(timeout_s=1.0)
                except Exception:
                    pass

        if self._roi_thread:
            self._roi_thread.join(timeout=5.0)

        self._log.info("Capture orchestrator stopped")

    def pause(self) -> None:
        self._paused.set()
        self._log.info("Capture orchestrator paused")

    def resume(self) -> None:
        self._paused.clear()
        self._log.info("Capture orchestrator resumed")

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    def _run_capture_loop(self) -> None:  # pragma: no cover - depends on Windows APIs
        min_interval = self._capture_config.hid.min_interval_ms / 1000
        interval = max(
            1.0 / max(self._capture_config.hid.fps_soft_cap, 0.01), min_interval
        )
        try:
            while self._running.is_set():
                loop_start = time.monotonic()
                now_ms = int(loop_start * 1000)
                active = (
                    now_ms < self._raw_input.active_until_ts
                ) and not self._paused.is_set()

                if active:
                    if not self._was_active:
                        try:
                            self._active_segment_id = self._start_segment()
                        except Exception as exc:
                            self._log.exception("Failed to start segment: {}", exc)
                            self._active_segment_id = None
                    try:
                        self._capture_tick()
                    except Exception as exc:
                        self._log.exception("Capture tick failed: {}", exc)
                        time.sleep(min(1.0, interval))
                    elapsed = time.monotonic() - loop_start
                    time.sleep(max(0.0, interval - elapsed))
                else:
                    if self._was_active:
                        try:
                            self._close_segment(self._active_segment_id)
                        except Exception as exc:
                            self._log.exception("Failed to close segment: {}", exc)
                        self._active_segment_id = None
                    time.sleep(0.1)

                self._was_active = active
        finally:
            segment_id = self._active_segment_id
            self._active_segment_id = None
            if segment_id:
                try:
                    self._close_segment(segment_id)
                except Exception as exc:
                    self._log.exception("Failed to close segment on shutdown: {}", exc)
            self._was_active = False

    def _capture_tick(self) -> None:
        frames = self._backend.grab_all()
        if not frames:
            return
        self._refresh_monitors()
        cursor_x, cursor_y = self._get_cursor_pos()
        monitor = self._find_monitor(cursor_x, cursor_y)
        ctx = get_foreground_context()
        foreground_process = ctx.process_name if ctx else "unknown"
        foreground_window = ctx.window_title if ctx else "unknown"
        is_fullscreen = False
        if ctx and ctx.hwnd:
            is_fullscreen = is_fullscreen_window(ctx.hwnd)
        if is_fullscreen and self._capture_config.hid.block_fullscreen:
            return
        if monitor and monitor.id in frames:
            roi = self._crop_roi(frames[monitor.id], monitor, cursor_x, cursor_y)
            duplicate = self._duplicate_detector.update(
                Image.fromarray(np.ascontiguousarray(roi))
            )
            observation_id = str(uuid4())
            if not duplicate.is_duplicate:
                roi_item = ROIItem(
                    image=roi,
                    captured_at=dt.datetime.now(dt.timezone.utc),
                    observation_id=observation_id,
                    segment_id=self._active_segment_id,
                    cursor_x=cursor_x,
                    cursor_y=cursor_y,
                    monitor_id=monitor.id,
                    foreground_process=foreground_process or "unknown",
                    foreground_window=foreground_window or "unknown",
                    is_fullscreen=is_fullscreen,
                )
                self._enqueue_roi(roi_item)
                captures_taken_total.inc()
        self._enqueue_video(
            frames, foreground_process, foreground_window, is_fullscreen
        )

    def _enqueue_roi(self, item: ROIItem) -> None:
        if self._should_throttle_ocr():
            captures_skipped_backpressure_total.inc()
            self._apply_backpressure("ocr_backlog")
            return
        try:
            self._roi_queue.put(item, timeout=0.1)
            roi_queue_depth.set(self._roi_queue.qsize())
            self._backoff_s = 0.0
        except queue.Full:
            roi_queue_full_total.inc()
            self._apply_backpressure("roi_queue_full")

    def _enqueue_video(
        self,
        frames: Dict[str, np.ndarray],
        foreground_process: str,
        foreground_window: str,
        is_fullscreen: bool,
    ) -> None:
        if not self._segment_recorder.is_available:
            return
        capture_frames: list[CaptureFrame] = []
        for monitor in self._monitors:
            frame = frames.get(monitor.id)
            if frame is None:
                continue
            image = Image.fromarray(np.ascontiguousarray(frame))
            capture_frames.append(
                CaptureFrame(
                    timestamp=dt.datetime.now(dt.timezone.utc),
                    image=image,
                    foreground_process=foreground_process or "unknown",
                    foreground_window=foreground_window or "unknown",
                    monitor_id=monitor.id,
                    monitor_bounds=(
                        monitor.left,
                        monitor.top,
                        monitor.width,
                        monitor.height,
                    ),
                    is_fullscreen=is_fullscreen,
                )
            )
        if capture_frames:
            now = time.monotonic()
            if now < self._video_disabled_until:
                video_disabled.set(1)
                return
            video_disabled.set(0)
            self._video_tick += 1
            if (
                self._video_sampling_divisor > 1
                and self._video_tick % self._video_sampling_divisor != 0
            ):
                return
            accepted = self._segment_recorder.enqueue(capture_frames)
            if not accepted:
                video_backpressure_events_total.inc()
                self._video_last_drop = now
                self._video_drop_window.append(now)
                while (
                    self._video_drop_window
                    and now - self._video_drop_window[0] > self._video_drop_window_s
                ):
                    self._video_drop_window.popleft()
                self._video_sampling_divisor = min(self._video_sampling_divisor * 2, 16)
                if len(self._video_drop_window) >= 10:
                    self._video_disabled_until = now + 10.0
                    self._video_drop_window.clear()
                    self._log.warning(
                        "Video capture disabled for cooldown due to backpressure"
                    )
            else:
                if (
                    self._video_sampling_divisor > 1
                    and now - self._video_last_drop > 5.0
                ):
                    self._video_sampling_divisor = max(
                        1, self._video_sampling_divisor - 1
                    )

    def _run_roi_saver(self) -> None:
        while self._running.is_set() or not self._roi_queue.empty():
            try:
                item = self._roi_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                path = self._save_roi(item)
                if path is None:
                    self._persist_skipped_capture(item, "roi_write_failed")
                    continue
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
                    "Failed to persist ROI observation {}: {}",
                    item.observation_id,
                    exc,
                )
            finally:
                self._roi_queue.task_done()
                roi_queue_depth.set(self._roi_queue.qsize())

    def _save_roi(self, item: ROIItem) -> Optional[Path]:
        if not self._has_disk_space():
            disk_low_total.inc()
            self._apply_backpressure("disk_low")
            return None
        if self._media_store:
            return self._media_store.write_roi(
                item.image, item.captured_at, item.observation_id
            )
        timestamp = item.captured_at.astimezone(dt.timezone.utc)
        path = (
            Path(self._capture_config.data_dir)
            / "media"
            / "roi"
            / timestamp.strftime("%Y/%m/%d")
            / f"{timestamp.strftime('%H%M%S_%f')}_{item.observation_id}.webp"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(np.ascontiguousarray(item.image))
        image.save(path, format="WEBP", lossless=True, quality=100, method=6)
        return path

    def _persist_observation(self, item: ROIItem, path: Path) -> None:
        def _write(session) -> None:
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
                    foreground_process=item.foreground_process or "unknown",
                    foreground_window=item.foreground_window or "unknown",
                    monitor_id=item.monitor_id,
                    is_fullscreen=item.is_fullscreen,
                    ocr_status="pending",
                )
            )

        self._database.transaction(_write)

    def _persist_skipped_capture(self, item: ROIItem, reason: str) -> None:
        def _write(session) -> None:
            session.add(
                CaptureRecord(
                    id=item.observation_id,
                    captured_at=item.captured_at,
                    image_path=None,
                    foreground_process=item.foreground_process or "unknown",
                    foreground_window=item.foreground_window or "unknown",
                    monitor_id=item.monitor_id,
                    is_fullscreen=item.is_fullscreen,
                    ocr_status="skipped",
                    ocr_last_error=reason,
                )
            )

        self._database.transaction(_write)

    def _start_segment(self) -> str:
        segment_id = str(uuid4())
        started_at = dt.datetime.now(dt.timezone.utc)
        self._database.transaction(
            lambda session: session.add(
                SegmentRecord(
                    id=segment_id,
                    started_at=started_at,
                    state="recording",
                )
            )
        )
        output_path = None
        if self._media_store:
            staging_path, final_path = self._media_store.reserve_video_paths(
                started_at, segment_id
            )
            output_path = staging_path
            self._segment_video_paths[segment_id] = final_path
        self._segment_recorder.start_segment(
            started_at=started_at,
            segment_id=segment_id,
            output_path=output_path,
        )
        return segment_id

    def _close_segment(self, segment_id: Optional[str]) -> None:
        if not segment_id:
            return
        segment = self._segment_recorder.stop_segment()

        def _write(session) -> None:
            record = session.get(SegmentRecord, segment_id)
            if record:
                record.ended_at = dt.datetime.now(dt.timezone.utc)
                if segment is None:
                    record.state = "closed_no_video"
                else:
                    record.state = segment.state or "closed"
                    final_path = None
                    if segment.video_path and self._media_store:
                        final_path = self._media_store.finalize_video(
                            Path(segment.video_path),
                            self._segment_video_paths.get(
                                segment_id, Path(segment.video_path)
                            ),
                        )
                    record.video_path = (
                        str(final_path or segment.video_path)
                        if segment.video_path
                        else None
                    )
                    record.encoder = segment.encoder
                    record.frame_count = segment.frame_count

        self._database.transaction(_write)
        self._segment_video_paths.pop(segment_id, None)
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
            self._log.warning("DxCam unavailable; falling back to MSS: {}", exc)
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
                    .where(
                        CaptureRecord.ocr_status.in_(["pending", "processing"]),
                        CaptureRecord.ocr_attempts
                        < self._worker_config.ocr_max_attempts,
                    )
                ).scalar_one()
                lease_cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
                    milliseconds=self._worker_config.ocr_lease_ms
                )
                stale = session.execute(
                    select(func.count())
                    .select_from(CaptureRecord)
                    .where(
                        CaptureRecord.ocr_status == "processing",
                        func.coalesce(
                            CaptureRecord.ocr_heartbeat_at, CaptureRecord.ocr_started_at
                        )
                        < lease_cutoff,
                    )
                ).scalar_one()
        except Exception as exc:  # pragma: no cover - defensive
            self._log.debug("Failed to check OCR backlog: {}", exc)
            self._ocr_backlog_cached_throttle = False
            return False
        self._ocr_backlog_last_count = int(count or 0)
        ocr_backlog.set(self._ocr_backlog_last_count)
        ocr_backlog_gauge.set(self._ocr_backlog_last_count)
        ocr_stale_processing_gauge.set(int(stale or 0))
        self._ocr_backlog_cached_throttle = (
            self._ocr_backlog_last_count >= self._ocr_backlog_soft_limit
        )
        if self._ocr_backlog_cached_throttle and now - self._ocr_backlog_last_log > 5.0:
            self._ocr_backlog_last_log = now
            self._log.warning(
                "OCR backlog at {} (limit {}); throttling capture",
                self._ocr_backlog_last_count,
                self._ocr_backlog_soft_limit,
            )
        return self._ocr_backlog_cached_throttle

    def _apply_backpressure(self, reason: str) -> None:
        self._backoff_s = min(2.0, self._backoff_s * 2 or 0.25)
        if self._backoff_s > 0:
            self._log.debug(
                "Backpressure ({}): sleeping for {:.2f}s", reason, self._backoff_s
            )
            time.sleep(self._backoff_s)

    def _has_disk_space(self) -> bool:
        try:
            staging = shutil.disk_usage(self._capture_config.staging_dir)
            data = shutil.disk_usage(self._capture_config.data_dir)
        except FileNotFoundError:
            return False
        min_staging = self._capture_config.staging_min_free_mb * 1024 * 1024
        min_data = self._capture_config.data_min_free_mb * 1024 * 1024
        return staging.free >= min_staging and data.free >= min_data

    def _refresh_monitors(self) -> None:
        backend_monitors = self._backend.monitors
        if {m.id for m in backend_monitors} != {m.id for m in self._monitors}:
            self._monitors = backend_monitors
