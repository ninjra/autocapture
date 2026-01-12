"""Resource path helpers for bundled assets."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .config import FFmpegConfig, QdrantConfig


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False) or getattr(sys, "_MEIPASS", None))


def resource_root() -> Path:
    if getattr(sys, "_MEIPASS", None):  # pragma: no cover - runtime bundle
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def app_local_data_dir(app_name: str = "Autocapture") -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / app_name
    return Path.home() / "AppData" / "Local" / app_name


def default_data_dir() -> Path:
    if sys.platform == "win32":
        return app_local_data_dir() / "data"
    return Path("./data")


def default_staging_dir() -> Path:
    if sys.platform == "win32":
        return app_local_data_dir() / "staging"
    return Path("./staging")


def default_config_path() -> Path:
    if sys.platform == "win32":
        return app_local_data_dir() / "autocapture.yml"
    return Path("autocapture.yml")


def ensure_config_path(config_path: Path) -> Path:
    if config_path.exists():
        return config_path
    template = resource_root() / "autocapture.yml"
    if template.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


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


def find_bundled_qdrant(config: QdrantConfig) -> Path:
    root = resource_root()
    candidates = [root / "qdrant" / "qdrant.exe"]
    exe_path = Path(sys.executable)
    candidates.append(exe_path.parent / "qdrant" / "qdrant.exe")

    if config.binary_path:
        candidates.insert(0, Path(config.binary_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Bundled qdrant.exe not found. Ensure Qdrant is bundled alongside the app."
    )


def resolve_qdrant_path(config: QdrantConfig) -> Path | None:
    try:
        return find_bundled_qdrant(config)
    except FileNotFoundError:
        return None
