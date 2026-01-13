"""Preset definitions for privacy-first and high-fidelity modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AppConfig


@dataclass(frozen=True)
class Preset:
    name: str
    description: str


PRIVACY_FIRST = Preset(
    name="privacy_first",
    description="Privacy-first (recommended) preset with minimized capture footprint.",
)

HIGH_FIDELITY = Preset(
    name="high_fidelity",
    description="High-fidelity preset with richer capture defaults.",
)

PRESETS = {
    PRIVACY_FIRST.name: PRIVACY_FIRST,
    HIGH_FIDELITY.name: HIGH_FIDELITY,
}


def apply_preset(config: "AppConfig", preset_name: str) -> "AppConfig":
    """Apply a preset to the runtime config in-place."""

    preset = PRESETS.get(preset_name)
    if not preset:
        return config

    if preset.name == PRIVACY_FIRST.name:
        config.capture.record_video = False
        config.capture.always_store_fullres = False
        config.capture.hid.fps_soft_cap = 0.5
        config.retention.roi_days = 7
        config.retention.screenshot_ttl_days = 30
        config.privacy.sanitize_default = True
        config.privacy.extractive_only_default = True
    elif preset.name == HIGH_FIDELITY.name:
        config.capture.record_video = True
        config.capture.always_store_fullres = True
        config.capture.hid.fps_soft_cap = max(config.capture.hid.fps_soft_cap, 2.0)
    return config
