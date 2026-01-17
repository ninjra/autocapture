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

from .privacy import PrivacyPolicy, get_screen_lock_status
from .privacy_filter import apply_exclude_region_masks, should_skip_capture
from ..config import CaptureConfig, FFmpegConfig, PrivacyConfig, WorkerConfig
from ..runtime_governor import RuntimeGovernor, RuntimeMode
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
class CaptureItem:
    full_image: np.ndarray
    focus_image: np.ndarray | None
    captured_at: dt.datetime
    capture_id: str
    segment_id: Optional[str]
    cursor_x: int
    cursor_y: int
    monitor_id: str
    foreground_process: str
    foreground_window: str
    is_fullscreen: bool

    @property
    def image(self) -> np.ndarray:
        return self.focus_image if self.focus_image is not None else self.full_image


class CaptureOrchestrator:
    """Coordinate raw input gating, multi-monitor capture, and ROI persistence."""

    def __init__(
        self,
        database: DatabaseManager,
        capture_config: CaptureConfig,
        worker_config: WorkerConfig,
        privacy_config: PrivacyConfig | None = None,
        roi_w: int | None = None,
        roi_h: int | None = None,
        roi_queue_size: int = 256,
        on_ocr_observation: Optional[Callable[[str], None]] = None,
        on_vision_observation: Optional[Callable[[str], None]] = None,
        vision_sample_rate: float = 0.0,
        on_segment_finalize: Optional[Callable[[str], None]] = None,
        hotkey_callback: Optional[Callable[[], None]] = None,
        raw_input: RawInputListener | None = None,
        ocr_backlog_check_s: float = 1.0,
        media_store: MediaStore | None = None,
        ffmpeg_config: FFmpegConfig | None = None,
        backend: object | None = None,
        runtime_governor: RuntimeGovernor | None = None,
        runtime_auto_pause: bool | None = None,
    ) -> None:
        self._log = get_logger("orchestrator")
        self._database = database
        self._capture_config = capture_config
        self._worker_config = worker_config
        self._privacy_config = privacy_config or PrivacyConfig()
        self._privacy = self._privacy_config
        self._privacy_policy = PrivacyPolicy(self._privacy)
        focus_size = capture_config.focus_crop_size or capture_config.tile_size
        self._roi_w = roi_w if roi_w is not None else focus_size
        self._roi_h = roi_h if roi_h is not None else focus_size
        self._fullscreen_primary = capture_config.fullscreen_primary
        self._fullscreen_width = capture_config.fullscreen_width
        self._focus_crop_enabled = capture_config.focus_crop_enabled
        self._focus_reference = capture_config.focus_crop_reference
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
        self._roi_queue: queue.Queue[CaptureItem] = queue.Queue(maxsize=roi_queue_size)
        self._segment_recorder = SegmentRecorder(
            capture_config=capture_config,
            ffmpeg_config=ffmpeg_config or FFmpegConfig(),
        )
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
        self._auto_paused = threading.Event()
        self._auto_pause_reason: str | None = None
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
        self._capture_restart_attempts = 0
        self._capture_restart_limit = 5
        self._capture_restart_backoff_s = 0.5
        self._capture_restart_max_backoff_s = 30.0
        self._runtime = runtime_governor
        self._runtime_auto_pause = bool(runtime_auto_pause)

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

    def _set_auto_pause(self, reason: str) -> None:
        if not self._auto_paused.is_set() or self._auto_pause_reason != reason:
            self._auto_pause_reason = reason
            self._auto_paused.set()
            self._log.info("Capture auto-paused (%s)", reason)

    def _clear_auto_pause(self) -> None:
        if self._auto_paused.is_set():
            self._auto_paused.clear()
            self._auto_pause_reason = None
            self._log.info("Capture auto-pause cleared")

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set() or self._auto_paused.is_set()

    def _run_capture_loop(self) -> None:  # pragma: no cover - depends on Windows APIs
        while self._running.is_set():
            try:
                self._run_capture_loop_once()
                if not self._running.is_set():
                    return
                self._capture_restart_attempts += 1
                self._log.warning("Capture loop exited unexpectedly; restarting.")
            except Exception as exc:
                self._capture_restart_attempts += 1
                self._log.exception("Capture loop crashed: {}", exc)
            if self._capture_restart_attempts >= self._capture_restart_limit:
                self._log.error(
                    "Capture loop failed %s times; giving up to avoid crash loop.",
                    self._capture_restart_attempts,
                )
                self._running.clear()
                return
            delay = min(self._capture_restart_backoff_s, self._capture_restart_max_backoff_s)
            self._log.warning("Restarting capture loop in %.1fs", delay)
            time.sleep(delay)
            self._capture_restart_backoff_s = min(
                self._capture_restart_backoff_s * 2, self._capture_restart_max_backoff_s
            )

    def _run_capture_loop_once(self) -> None:  # pragma: no cover - depends on Windows APIs
        min_interval = self._capture_config.hid.min_interval_ms / 1000
        interval = max(1.0 / max(self._capture_config.hid.fps_soft_cap, 0.01), min_interval)
        try:
            while self._running.is_set():
                loop_start = time.monotonic()
                if (
                    self._runtime
                    and self._runtime.current_mode == RuntimeMode.FULLSCREEN_HARD_PAUSE
                ):
                    time.sleep(min(1.0, interval))
                    continue
                now_ms = int(loop_start * 1000)
                active = (
                    (now_ms < self._raw_input.active_until_ts)
                    and not self._paused.is_set()
                    and not self._auto_paused.is_set()
                )

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
        if (
            is_fullscreen
            and self._capture_config.hid.block_fullscreen
            and not self._runtime_auto_pause
        ):
            return
        if should_skip_capture(
            paused=self._privacy.paused,
            monitor_id=monitor.id if monitor else None,
            process_name=foreground_process,
            window_title=foreground_window,
            exclude_monitors=self._privacy.exclude_monitors,
            exclude_processes=self._privacy.exclude_processes,
            exclude_window_title_regex=self._privacy.exclude_window_title_regex,
        ):
            return
        screen_locked, secure_desktop = get_screen_lock_status()
        decision = self._privacy_policy.evaluate(
            ctx, screen_locked=screen_locked, secure_desktop=secure_desktop
        )
        if not decision.allowed:
            if decision.auto_pause:
                self._set_auto_pause(decision.reason or "privacy")
            return
        self._clear_auto_pause()
        if monitor and monitor.id in frames:
            roi, roi_origin = self._crop_roi(frames[monitor.id], monitor, cursor_x, cursor_y)
            apply_exclude_region_masks(
                roi,
                monitor_id=monitor.id,
                roi_origin_x=roi_origin[0],
                roi_origin_y=roi_origin[1],
                exclude_regions=self._privacy.exclude_regions,
            )
            duplicate = self._duplicate_detector.update(Image.fromarray(np.ascontiguousarray(roi)))
            capture_id = str(uuid4())
            if not duplicate.is_duplicate:
                captured_at = dt.datetime.now(dt.timezone.utc)
                if self._fullscreen_primary:
                    full_image, full_origin = self._compose_fullscreen(frames, monitor)
                    full_image = self._apply_fullscreen_masks(full_image, full_origin, frames)
                    full_image = self._resize_fullscreen(full_image)
                else:
                    full_image = roi.copy()
                focus_image = roi if self._focus_crop_enabled else None
                capture_item = CaptureItem(
                    full_image=full_image,
                    focus_image=focus_image,
                    captured_at=captured_at,
                    capture_id=capture_id,
                    segment_id=self._active_segment_id,
                    cursor_x=cursor_x,
                    cursor_y=cursor_y,
                    monitor_id=monitor.id,
                    foreground_process=foreground_process or "unknown",
                    foreground_window=foreground_window or "unknown",
                    is_fullscreen=is_fullscreen,
                )
                self._enqueue_roi(capture_item)
                captures_taken_total.inc()
        self._enqueue_video(frames, foreground_process, foreground_window, is_fullscreen)

    def _enqueue_roi(self, item: CaptureItem) -> None:
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
                    self._log.warning("Video capture disabled for cooldown due to backpressure")
            else:
                if self._video_sampling_divisor > 1 and now - self._video_last_drop > 5.0:
                    self._video_sampling_divisor = max(1, self._video_sampling_divisor - 1)

    def _run_roi_saver(self) -> None:
        while self._running.is_set() or not self._roi_queue.empty():
            try:
                item = self._roi_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                full_path = self._save_fullscreen(item)
                if full_path is None:
                    self._persist_skipped_capture(item, "fullscreen_write_failed")
                    continue
                focus_path = None
                if item.focus_image is not None and self._focus_crop_enabled:
                    focus_path = self._save_focus(item)
                    if focus_path is None:
                        self._log.warning("Focus crop write failed for {}", item.capture_id)
                self._persist_capture(item, full_path, focus_path)
                if self._on_ocr_observation:
                    self._on_ocr_observation(item.capture_id)
                if (
                    self._on_vision_observation
                    and self._vision_sample_rate > 0
                    and random.random() < self._vision_sample_rate
                ):
                    self._on_vision_observation(item.capture_id)
            except Exception as exc:  # pragma: no cover - side effects
                self._log.exception(
                    "Failed to persist capture {}: {}",
                    item.capture_id,
                    exc,
                )
            finally:
                self._roi_queue.task_done()
                roi_queue_depth.set(self._roi_queue.qsize())

    def _save_fullscreen(self, item: CaptureItem) -> Optional[Path]:
        if not self._has_disk_space():
            disk_low_total.inc()
            self._apply_backpressure("disk_low")
            return None
        if self._media_store:
            return self._media_store.write_fullscreen(
                item.full_image, item.captured_at, item.capture_id
            )
        timestamp = item.captured_at.astimezone(dt.timezone.utc)
        path = (
            Path(self._capture_config.data_dir)
            / "media"
            / "screen"
            / timestamp.strftime("%Y/%m/%d")
            / f"{timestamp.strftime('%H%M%S_%f')}_{item.capture_id}.webp"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(np.ascontiguousarray(item.full_image))
        image.save(path, format="WEBP", lossless=True, quality=100, method=6)
        return path

    def _save_focus(self, item: CaptureItem) -> Optional[Path]:
        if item.focus_image is None:
            return None
        if not self._has_disk_space():
            disk_low_total.inc()
            self._apply_backpressure("disk_low")
            return None
        if self._media_store:
            return self._media_store.write_roi(item.focus_image, item.captured_at, item.capture_id)
        timestamp = item.captured_at.astimezone(dt.timezone.utc)
        path = (
            Path(self._capture_config.data_dir)
            / "media"
            / "roi"
            / timestamp.strftime("%Y/%m/%d")
            / f"{timestamp.strftime('%H%M%S_%f')}_{item.capture_id}.webp"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(np.ascontiguousarray(item.focus_image))
        image.save(path, format="WEBP", lossless=True, quality=100, method=6)
        return path

    def _persist_capture(self, item: CaptureItem, full_path: Path, focus_path: Path | None) -> None:
        def _write(session) -> None:
            if focus_path is not None:
                session.add(
                    ObservationRecord(
                        id=item.capture_id,
                        captured_at=item.captured_at,
                        image_path=str(focus_path),
                        segment_id=item.segment_id,
                        cursor_x=item.cursor_x,
                        cursor_y=item.cursor_y,
                        monitor_id=item.monitor_id,
                    )
                )
            session.add(
                CaptureRecord(
                    id=item.capture_id,
                    captured_at=item.captured_at,
                    image_path=str(full_path),
                    focus_path=str(focus_path) if focus_path else None,
                    foreground_process=item.foreground_process or "unknown",
                    foreground_window=item.foreground_window or "unknown",
                    monitor_id=item.monitor_id,
                    is_fullscreen=item.is_fullscreen,
                    ocr_status="pending",
                )
            )

        self._database.transaction(_write)

    def _persist_skipped_capture(self, item: CaptureItem, reason: str) -> None:
        def _write(session) -> None:
            session.add(
                CaptureRecord(
                    id=item.capture_id,
                    captured_at=item.captured_at,
                    image_path=None,
                    focus_path=None,
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
            staging_path, final_path = self._media_store.reserve_video_paths(started_at, segment_id)
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
                            self._segment_video_paths.get(segment_id, Path(segment.video_path)),
                        )
                    record.video_path = (
                        str(final_path or segment.video_path) if segment.video_path else None
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
    ) -> tuple[np.ndarray, tuple[int, int]]:
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
        return frame[y0:y1, x0:x1], (x0, y0)

    def _compose_fullscreen(
        self, frames: Dict[str, np.ndarray], active_monitor: MonitorInfo | None
    ) -> tuple[np.ndarray, tuple[int, int]]:
        if (
            self._capture_config.layout_mode == "per_monitor"
            and active_monitor
            and active_monitor.id in frames
        ):
            return np.ascontiguousarray(frames[active_monitor.id]), (
                active_monitor.left,
                active_monitor.top,
            )
        monitors = [monitor for monitor in self._monitors if monitor.id in frames]
        if not monitors:
            first = next(iter(frames.values()))
            return np.ascontiguousarray(first), (0, 0)
        left = min(monitor.left for monitor in monitors)
        top = min(monitor.top for monitor in monitors)
        right = max(monitor.left + monitor.width for monitor in monitors)
        bottom = max(monitor.top + monitor.height for monitor in monitors)
        canvas = np.zeros((bottom - top, right - left, 3), dtype=np.uint8)
        for monitor in monitors:
            frame = frames.get(monitor.id)
            if frame is None:
                continue
            x0 = monitor.left - left
            y0 = monitor.top - top
            canvas[y0 : y0 + monitor.height, x0 : x0 + monitor.width] = frame
        return canvas, (left, top)

    def _apply_fullscreen_masks(
        self, image: np.ndarray, origin: tuple[int, int], frames: Dict[str, np.ndarray]
    ) -> np.ndarray:
        if not self._privacy.exclude_regions:
            return image
        origin_x, origin_y = origin
        for monitor in self._monitors:
            if monitor.id not in frames:
                continue
            apply_exclude_region_masks(
                image,
                monitor_id=monitor.id,
                roi_origin_x=monitor.left - origin_x,
                roi_origin_y=monitor.top - origin_y,
                exclude_regions=self._privacy.exclude_regions,
            )
        return image

    def _resize_fullscreen(self, image: np.ndarray) -> np.ndarray:
        target_width = int(self._fullscreen_width or 0)
        if target_width <= 0:
            return image
        height, width = image.shape[:2]
        if width <= target_width:
            return image
        scale = target_width / float(width)
        target_height = max(1, int(round(height * scale)))
        resized = Image.fromarray(np.ascontiguousarray(image)).resize(
            (target_width, target_height),
            resample=Image.LANCZOS,
        )
        return np.array(resized)

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
                        CaptureRecord.ocr_attempts < self._worker_config.ocr_max_attempts,
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
                        func.coalesce(CaptureRecord.ocr_heartbeat_at, CaptureRecord.ocr_started_at)
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
            self._log.debug("Backpressure ({}): sleeping for {:.2f}s", reason, self._backoff_s)
            time.sleep(self._backoff_s)

    def _has_disk_space(self) -> bool:
        try:
            staging = shutil.disk_usage(self._capture_config.staging_dir)
            data = shutil.disk_usage(self._capture_config.data_dir)
        except FileNotFoundError:
            return False
        min_staging = self._capture_config.staging_min_free_mb * 1024 * 1024
        min_data = self._capture_config.data_min_free_mb * 1024 * 1024
        if min_staging >= staging.total:
            min_staging = 0
        if min_data >= data.total:
            min_data = 0
        return staging.free >= min_staging and data.free >= min_data

    def _refresh_monitors(self) -> None:
        backend_monitors = self._backend.monitors
        if {m.id for m in backend_monitors} != {m.id for m in self._monitors}:
            self._monitors = backend_monitors
