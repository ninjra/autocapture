from pathlib import Path

from autocapture.config import load_config


def test_ffmpeg_not_require_bundled_in_test_mode(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "autocapture.yml"
    config_path.write_text("ffmpeg:\\n  require_bundled: true\\n", encoding="utf-8")
    monkeypatch.setenv("AUTOCAPTURE_TEST_MODE", "1")
    config = load_config(config_path)
    assert config.ffmpeg.require_bundled is False
