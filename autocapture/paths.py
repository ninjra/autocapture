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

    raise FileNotFoundError(
        "Bundled ffmpeg.exe not found. Ensure ffmpeg is bundled alongside the app."
    )


def resolve_ffmpeg_path(config: FFmpegConfig) -> Path | None:
    if not config.enabled:
        return None
    if config.require_bundled:
        try:
            return find_bundled_ffmpeg(config)
        except FileNotFoundError as exc:
            if config.allow_disable:
                return None
            raise FileNotFoundError(
                f"{exc} Set ffmpeg.require_bundled=false to allow fallbacks."
            ) from exc

    bundled_path = None
    try:
        bundled_path = find_bundled_ffmpeg(config)
    except FileNotFoundError:
        bundled_path = None

    if bundled_path:
        return bundled_path
    if config.explicit_path:
        explicit = Path(config.explicit_path)
        if explicit.exists():
            return explicit
        if not config.allow_disable:
            raise FileNotFoundError(
                f"ffmpeg explicit_path not found: {explicit}"
            )
    if config.allow_system:
        from shutil import which

        resolved = which("ffmpeg")
        if resolved:
            return Path(resolved)
    if config.allow_disable:
        return None
    raise FileNotFoundError(
        "ffmpeg binary not found. Bundle ffmpeg, set ffmpeg.explicit_path, "
        "or allow PATH lookup by setting ffmpeg.allow_system=true."
    )
