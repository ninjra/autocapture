"""Resource path helpers for bundled assets."""

from __future__ import annotations

import sys
from pathlib import Path

from .config import FFmpegConfig


def resource_root() -> Path:
    if getattr(sys, "_MEIPASS", None):  # pragma: no cover - runtime bundle
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def find_bundled_ffmpeg(config: FFmpegConfig) -> Path:
    root = resource_root()
    candidates = [root / candidate for candidate in config.relative_path_candidates]

    exe_path = Path(sys.executable)
    candidates.append(exe_path.parent / "ffmpeg" / "bin" / "ffmpeg.exe")
    candidates.append(exe_path.parent / "ffmpeg" / "ffmpeg.exe")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if config.enabled:
        raise FileNotFoundError(
            "Bundled ffmpeg.exe not found. Ensure ffmpeg is bundled alongside the app."
        )
    raise FileNotFoundError("ffmpeg disabled")
