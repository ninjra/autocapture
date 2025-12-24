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
from uuid import uuid4

from PIL import Image

from ..config import CaptureConfig
from ..logging_utils import get_logger
from .screen_capture import CaptureFrame


@dataclass(slots=True)
class VideoSegment:
    id: str
    started_at: dt.datetime
    video_path: Optional[Path]
    state: str
    encoder: Optional[str] = None


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
        ffmpeg_path = Path(path_dir) / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
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

    def __init__(self, config: CaptureConfig) -> None:
        self._config = config
        self._log = get_logger("recorder")
        self._ffmpeg_path = find_ffmpeg(config.data_dir)
        if self._ffmpeg_path is None:
            self._log.warning("FFmpeg not found; video recording disabled.")
        self._queue: queue.Queue[list[CaptureFrame] | None] = queue.Queue(
            maxsize=1024
        )
        self._thread: Optional[threading.Thread] = None
        self._segment: Optional[VideoSegment] = None
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._frame_size: Optional[tuple[int, int]] = None

    def start_segment(self, started_at: dt.datetime) -> Optional[VideoSegment]:
        if not self._config.record_video:
            return None
        if self._ffmpeg_path is None:
            return None
        if self._segment is not None and self._segment.state == "recording":
            return self._segment
        if self._segment is not None and self._segment.state != "recording":
            self._segment = None

        segment_id = str(uuid4())
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
        self._queue = queue.Queue(maxsize=1024)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._segment

    def enqueue(self, frame: CaptureFrame) -> None:
        if self._segment is None or self._segment.state != "recording":
            return
        try:
            self._queue.put_nowait([frame])
        except queue.Full:
            self._log.warning("Dropping video frame due to recorder backpressure.")

    def stop_segment(self, timeout_s: float = 5.0) -> Optional[VideoSegment]:
        if self._segment is None:
            return None
        self._queue.put(None)
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
            item = self._queue.get()
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
                        self._segment.state = "disabled"
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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            self._log.warning("Failed to launch ffmpeg: {}", exc)
            return None

        time.sleep(0.25)
        if process.poll() is not None:
            stderr_output = b""
            if process.stderr:
                stderr_output = process.stderr.read() or b""
            stderr_text = stderr_output.decode("utf-8", errors="ignore")
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
        except BrokenPipeError:
            self._log.warning("FFmpeg pipe closed unexpectedly.")
            self._terminate_process()

    def _finalize(self) -> None:
        if self._process is None:
            return
        if self._process.stdin:
            try:
                self._process.stdin.close()
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            return_code = self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._log.warning("FFmpeg did not exit in time; terminating.")
            self._terminate_process()
            return_code = -1
        if self._segment:
            self._segment.state = "completed" if return_code == 0 else "failed"
        self._process = None
        self._frame_size = None

    def _terminate_process(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
