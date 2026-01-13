from __future__ import annotations

from autocapture.config import AppConfig
from autocapture.presets import apply_preset


def test_privacy_first_preset_adjusts_config() -> None:
    config = AppConfig()
    apply_preset(config, "privacy_first")
    assert config.capture.record_video is False
    assert config.capture.always_store_fullres is False
    assert config.capture.hid.fps_soft_cap <= 1.0
    assert config.retention.roi_days <= 30
    assert config.retention.screenshot_ttl_days <= 30
