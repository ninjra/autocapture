import os

from autocapture.config import FFmpegConfig
from autocapture.paths import resolve_ffmpeg_path


def test_resolve_ffmpeg_path_allows_missing_when_test_mode(monkeypatch):
    monkeypatch.setenv("AUTOCAPTURE_TEST_MODE", "1")
    config = FFmpegConfig(enabled=True, require_bundled=True, allow_disable=False)
    path = resolve_ffmpeg_path(config)
    assert path is None or path.exists()
    monkeypatch.delenv("AUTOCAPTURE_TEST_MODE", raising=False)
