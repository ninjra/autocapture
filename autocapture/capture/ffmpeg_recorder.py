"""FFmpeg-backed video recorder for activity segments."""

from __future__ import annotations

import datetime as dt
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from collections import deque

from PIL import Image

from ..config import CaptureConfig
from ..logging_utils import get_logger
from ..observability.metrics import video_frames_dropped_total
from .screen_capture import CaptureFrame


@dataclass(slots=True)
class VideoSegment:
    id: str
    started_at: dt.datetime
    video_path: Optional[Path]
    state: str
    encoder: Optional[str] = None
    frame_count: int = 0
    error: Optional[str] = None


def find_ffmpeg(data_dir: Path) -> Optional[Path]:
    candidates: list[Path] = []
    if sys.platform == "win32":
        candidates.append(data_dir / "bin" / "ffmpeg.exe")
    else:
        candidates.append(data_dir / "bin" / "ffmpeg")

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        exe_dir = Path(getattr(sys, "_MEIPASS"))
        candidates.append(exe_dir / "ffmpeg.exe")
        candidates.append(exe_dir / "ffmpeg")

    exe_path = Path(sys.executable).resolve()
    candidates.append(exe_path.parent / "ffmpeg.exe")
    candidates.append(exe_path.parent / "ffmpeg")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        ffmpeg_path = Path(path_dir) / (
            "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        )
        if ffmpeg_path.exists():
            return ffmpeg_path

    return None


def _virtual_bounds(frames: Iterable[CaptureFrame]) -> tuple[int, int, int, int]:
    bounds = [frame.monitor_bounds for frame in frames]
    min_left = min(bound[0] for bound in bounds)
    min_top = min(bound[1] for bound in bounds)
    max_right = max(bound[0] + bound[2] for bound in bounds)
    max_bottom = max(bound[1] + bound[3] for bound in bounds)
    return min_left, min_top, max_right - min_left, max_bottom - min_top


def composite_frames(
    frames: list[CaptureFrame], layout_mode: str
) -> Optional[Image.Image]:
    if not frames:
        return None

    if layout_mode == "per_monitor":
        return frames[0].image.convert("RGB")

    left, top, width, height = _virtual_bounds(frames)
    canvas = Image.new("RGB", (width, height))
    for frame in frames:
        frame_left, frame_top, _, _ = frame.monitor_bounds
        offset = (frame_left - left, frame_top - top)
        canvas.paste(frame.image.convert("RGB"), offset)
    return canvas


class SegmentRecorder:
    """Manage a single FFmpeg process per active activity segment."""

    def __init__(self, capture_config: CaptureConfig) -> None:
        self._config = capture_config
        self._log = get_logger("recorder")
        self._ffmpeg_path = find_ffmpeg(self._config.data_dir)
        if self._ffmpeg_path is None:
            self._log.warning("FFmpeg not found; video recording disabled.")
        self._queue_maxsize = 512
        self._queue: queue.Queue[list[CaptureFrame] | None] = queue.Queue(
            maxsize=self._queue_maxsize
        )
        self._thread: Optional[threading.Thread] = None
        self._segment: Optional[VideoSegment] = None
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._frame_size: Optional[tuple[int, int]] = None
        self._stop_event = threading.Event()
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._dropped_frames = 0
        self._last_drop_log = 0.0
        self._drop_window_s = 10.0
        self._drop_window: deque[float] = deque(maxlen=500)

    @property
    def is_available(self) -> bool:
        return self._config.record_video and self._ffmpeg_path is not None

    def start_segment(
        self,
        started_at: dt.datetime,
        segment_id: str,
        output_path: Optional[Path] = None,
    ) -> Optional[VideoSegment]:
        if not self._config.record_video:
            return None
        if self._ffmpeg_path is None:
            return None
        if self._segment is not None and self._segment.state == "recording":
            return self._segment
        if self._segment is not None and self._segment.state != "recording":
            self._segment = None

        if output_path is None:
            date_prefix = started_at.strftime("%Y/%m/%d")
            output_dir = self._config.data_dir / "media" / "video" / date_prefix
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"segment_{segment_id}.mp4"

        self._segment = VideoSegment(
            id=segment_id,
            started_at=started_at,
            video_path=output_path,
            state="recording",
        )
        self._queue = queue.Queue(maxsize=self._queue_maxsize)
        self._stop_event.clear()
        self._stderr_lines.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._segment

    def enqueue(self, frames: list[CaptureFrame]) -> bool:
        if self._segment is None or self._segment.state != "recording":
            return False
        try:
            self._queue.put_nowait(frames)
            return True
        except queue.Full:
            self._dropped_frames += 1
            video_frames_dropped_total.inc()
            now = time.monotonic()
            self._drop_window.append(now)
            while (
                self._drop_window and now - self._drop_window[0] > self._drop_window_s
            ):
                self._drop_window.popleft()
            if now - self._last_drop_log > 5.0:
                self._last_drop_log = now
                self._log.warning(
                    "Dropping video frames due to recorder backpressure (dropped={}, recent={}).",
                    self._dropped_frames,
                    len(self._drop_window),
                )
            return False

    def stop_segment(self, timeout_s: float = 5.0) -> Optional[VideoSegment]:
        if self._segment is None:
            return None
        self._stop_event.set()
        self._signal_stop()
        if self._thread:
            self._thread.join(timeout=timeout_s)
        if self._thread and self._thread.is_alive():
            self._log.warning("Recorder thread timeout; forcing shutdown.")
            self._terminate_process()
        segment = self._segment
        if segment:
            self._log.info(
                "Segment {} finalized with state={} path={}",
                segment.id,
                segment.state,
                segment.video_path,
            )
        self._segment = None
        return segment

    def _run(self) -> None:
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                break
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            composite = composite_frames(item, self._config.layout_mode)
            if composite is None:
                continue
            if self._process is None:
                if self._segment is None:
                    break
                process, encoder = self._start_process(
                    composite.size, self._segment.video_path
                )
                if process is None:
                    if self._segment:
                        self._segment.state = "failed"
                        self._segment.error = "ffmpeg_failed_to_start"
                    break
                self._process = process
                if self._segment:
                    self._segment.encoder = encoder
                self._frame_size = composite.size
            if self._frame_size and composite.size != self._frame_size:
                self._log.warning(
                    "Frame size changed mid-segment ({} -> {}); dropping frame.",
                    self._frame_size,
                    composite.size,
                )
                continue
            self._write_frame(composite)

        self._finalize()

    def _start_process(
        self, size: tuple[int, int], output_path: Optional[Path]
    ) -> tuple[Optional[subprocess.Popen[bytes]], Optional[str]]:
        if self._ffmpeg_path is None or output_path is None:
            return None, None

        width, height = size
        fps = max(1, int(self._config.hid.fps_soft_cap))
        bitrate = self._config.video_bitrate
        preset = self._config.video_preset
        encoder_candidates = ["av1_nvenc", "hevc_nvenc", "h264_nvenc"]
        fallback_encoders = [("libx264", "ultrafast")]

        for encoder in encoder_candidates:
            process = self._spawn_ffmpeg(
                encoder=encoder,
                preset=preset,
                bitrate=bitrate,
                fps=fps,
                size=(width, height),
                output_path=output_path,
            )
            if process is None:
                continue
            self._log.info(
                "Video encoder selected: {} | bitrate={} | preset={}",
                encoder,
                bitrate,
                preset,
            )
            return process, encoder

        for encoder, fallback_preset in fallback_encoders:
            process = self._spawn_ffmpeg(
                encoder=encoder,
                preset=fallback_preset,
                bitrate=bitrate,
                fps=fps,
                size=(width, height),
                output_path=output_path,
            )
            if process is None:
                continue
            self._log.warning(
                "Falling back to CPU encoder: {} | bitrate={} | preset={}",
                encoder,
                bitrate,
                fallback_preset,
            )
            return process, encoder

        self._log.warning("No supported FFmpeg encoder found; video disabled.")
        return None, None

    def _spawn_ffmpeg(
        self,
        encoder: str,
        preset: str,
        bitrate: str,
        fps: int,
        size: tuple[int, int],
        output_path: Path,
    ) -> Optional[subprocess.Popen[bytes]]:
        width, height = size
        cmd = [
            str(self._ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            encoder,
            "-preset",
            preset,
            "-b:v",
            bitrate,
            "-movflags",
            "+faststart",
            "-y",
            str(output_path),
        ]
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            self._log.warning("Failed to launch ffmpeg: {}", exc)
            return None

        self._start_stderr_reader(process)
        time.sleep(0.25)
        if process.poll() is not None:
            stderr_text = self._drain_stderr_text()
            if "Unknown encoder" in stderr_text:
                self._log.warning("FFmpeg encoder unsupported: {}", encoder)
            else:
                self._log.warning("FFmpeg failed to start: {}", stderr_text.strip())
            return None

        return process

    def _write_frame(self, image: Image.Image) -> None:
        if self._process is None or self._process.stdin is None:
            return
        try:
            self._process.stdin.write(image.tobytes())
            if self._segment:
                self._segment.frame_count += 1
        except BrokenPipeError:
            self._log.warning("FFmpeg pipe closed unexpectedly.")
            if self._segment:
                self._segment.state = "failed"
                self._segment.error = "ffmpeg_broken_pipe"
            self._terminate_process()

    def _finalize(self) -> None:
        if self._process is None:
            if self._segment and self._segment.state == "recording":
                if self._segment.frame_count == 0:
                    self._segment.state = "closed_no_video"
                else:
                    self._segment.state = "failed"
                    if not self._segment.error:
                        self._segment.error = "ffmpeg_not_started"
            self._stop_event.clear()
            return
        return_code = self._shutdown_process()
        self._stop_stderr_reader()
        if self._segment:
            self._segment.state = "completed" if return_code == 0 else "failed"
            if return_code != 0 and not self._segment.error:
                self._segment.error = f"ffmpeg_exit_{return_code}"
                stderr_text = self._drain_stderr_text().strip()
                if stderr_text:
                    self._log.warning("FFmpeg stderr tail: {}", stderr_text)
        self._process = None
        self._frame_size = None
        self._stop_event.clear()

    def _terminate_process(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._stop_event.set()
        self._stop_stderr_reader()
        self._process = None

    def _shutdown_process(self) -> int:
        if self._process is None:
            return -1
        if self._process.stdin:
            try:
                self._process.stdin.close()
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            return self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._log.warning("FFmpeg did not exit in time; terminating.")
            self._process.terminate()
            try:
                return self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                return -1

    def _signal_stop(self) -> None:
        if self._queue.empty():
            try:
                self._queue.put_nowait(None)
                return
            except queue.Full:
                pass
        drained = 0
        while drained < 5:
            try:
                self._queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            self._log.warning(
                "Failed to enqueue recorder stop sentinel; forcing shutdown."
            )

    def _start_stderr_reader(self, process: subprocess.Popen[bytes]) -> None:
        if process.stderr is None:
            return

        def _drain() -> None:
            while not self._stop_event.is_set():
                line = process.stderr.readline()
                if not line:
                    break
                self._stderr_lines.append(line.decode("utf-8", errors="ignore"))

        self._stderr_thread = threading.Thread(target=_drain, daemon=True)
        self._stderr_thread.start()

    def _stop_stderr_reader(self) -> None:
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1.0)
        self._stderr_thread = None

    def _drain_stderr_text(self) -> str:
        return "".join(self._stderr_lines)
