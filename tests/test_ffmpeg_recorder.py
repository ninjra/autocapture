from __future__ import annotations

import datetime as dt
import os
import queue
import threading
import time
from pathlib import Path

from autocapture.capture.ffmpeg_recorder import SegmentRecorder, VideoSegment
from autocapture.config import CaptureConfig


def test_segment_recorder_constructs(tmp_path: Path) -> None:
    config = CaptureConfig(data_dir=tmp_path)
    recorder = SegmentRecorder(config)
    assert recorder is not None


def test_stop_segment_non_blocking_when_queue_full(tmp_path: Path) -> None:
    config = CaptureConfig(data_dir=tmp_path)
    recorder = SegmentRecorder(config)
    recorder._segment = VideoSegment(
        id="seg-1",
        started_at=dt.datetime.now(dt.timezone.utc),
        video_path=None,
        state="recording",
    )
    recorder._queue = queue.Queue(maxsize=1)
    recorder._queue.put_nowait([])

    start = time.monotonic()
    recorder.stop_segment(timeout_s=0.1)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0


def test_ffmpeg_stderr_is_drained(tmp_path: Path) -> None:
    config = CaptureConfig(data_dir=tmp_path)
    recorder = SegmentRecorder(config)
    rfd, wfd = os.pipe()
    read_handle = os.fdopen(rfd, "rb", closefd=True)

    class FakeProcess:
        stderr = read_handle

    process = FakeProcess()
    recorder._start_stderr_reader(process)

    def writer() -> None:
        with os.fdopen(wfd, "wb", closefd=True) as writer_handle:
            for idx in range(300):
                writer_handle.write(f"line-{idx}\n".encode("utf-8"))

    thread = threading.Thread(target=writer)
    thread.start()
    thread.join(timeout=2.0)

    recorder._stop_stderr_reader()

    assert recorder._stderr_lines
    assert len(recorder._stderr_lines) <= 200
