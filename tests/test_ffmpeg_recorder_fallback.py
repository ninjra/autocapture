import datetime as dt

from autocapture.capture.ffmpeg_recorder import SegmentRecorder, VideoSegment
from autocapture.config import CaptureConfig, FFmpegConfig


def _recorder(tmp_path) -> SegmentRecorder:
    capture = CaptureConfig(data_dir=tmp_path, staging_dir=tmp_path)
    ffmpeg = FFmpegConfig(allow_disable=True, allow_system=False, require_bundled=False)
    return SegmentRecorder(capture, ffmpeg)


def test_record_failure_disables_nvenc(monkeypatch, tmp_path) -> None:
    recorder = _recorder(tmp_path)
    recorder._segment = VideoSegment(
        id="seg",
        started_at=dt.datetime.now(dt.timezone.utc),
        video_path=None,
        state="recording",
        encoder="av1_nvenc",
    )
    monkeypatch.setattr("autocapture.capture.ffmpeg_recorder.time.monotonic", lambda: 100.0)
    recorder._record_failure("No capable devices found")
    assert recorder._force_cpu is True
    assert set(recorder._NVENC_ENCODERS).issubset(recorder._blocked_encoders)


def test_encoder_plan_uses_cpu_when_forced(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    recorder._force_cpu = True
    plan = recorder._encoder_plan("p4")
    assert plan
    assert all(not encoder.endswith("_nvenc") for encoder, _, _ in plan)
