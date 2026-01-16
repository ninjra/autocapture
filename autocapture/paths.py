"""Resource path helpers for bundled assets."""

from __future__ import annotations

import sys
import os
import ctypes
import uuid
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


def _split_path_parts(raw: str) -> list[str]:
    normalized = raw.replace("\\", "/")
    return [part for part in normalized.split("/") if part]


def _looks_like_windows_local_appdata(path: Path) -> bool:
    parts = [part.lower() for part in _split_path_parts(str(path))]
    for idx in range(len(parts) - 1):
        if parts[idx] == "appdata" and parts[idx + 1] == "local":
            return True
    return False


def _try_known_folder_local_appdata() -> Path | None:
    if sys.platform != "win32":
        return None
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    try:

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_uint32),
                ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        folder_id = uuid.UUID("{F1B32785-6FBA-4FCF-9D55-7B8E7F157091}")
        guid = GUID(
            folder_id.fields[0],
            folder_id.fields[1],
            folder_id.fields[2],
            (ctypes.c_ubyte * 8)(*folder_id.bytes[8:]),
        )
        path_ptr = ctypes.c_wchar_p()
        result = windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(guid), 0, None, ctypes.byref(path_ptr)
        )
        if result != 0:
            return None
        raw = path_ptr.value
        windll.ole32.CoTaskMemFree(path_ptr)
        if not raw:
            return None
        return Path(raw)
    except Exception:
        return None


def _resolve_windows_local_appdata_base() -> Path | None:
    candidates: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local))
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        candidates.append(Path(userprofile) / "AppData" / "Local")
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / ".." / "Local")
    homedrive = os.environ.get("HOMEDRIVE")
    homepath = os.environ.get("HOMEPATH")
    if homedrive and homepath:
        candidates.append(Path(f"{homedrive}{homepath}") / "AppData" / "Local")
    known_folder = _try_known_folder_local_appdata()
    if known_folder:
        candidates.append(known_folder)
    for candidate in candidates:
        if _looks_like_windows_local_appdata(candidate):
            return candidate
    home = Path.home()
    if ":" in str(home) or "\\" in str(home):
        fallback = home / "AppData" / "Local"
        if _looks_like_windows_local_appdata(fallback):
            return fallback
    system_drive = os.environ.get("SystemDrive") or "C:"
    fallback = Path(f"{system_drive}/Users/Public/AppData/Local")
    return fallback if _looks_like_windows_local_appdata(fallback) else None


def app_local_data_dir(app_name: str = "Autocapture") -> Path:
    if sys.platform == "win32":
        base = _resolve_windows_local_appdata_base()
        if base:
            return base / app_name
        fallback = Path.cwd() / "AppData" / "Local"
        return fallback / app_name
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
    candidates.extend(
        [
            root / "vendor" / "ffmpeg" / "bin" / "ffmpeg.exe",
            root / "vendor" / "ffmpeg" / "ffmpeg.exe",
        ]
    )

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
            raise FileNotFoundError(f"ffmpeg explicit_path not found: {explicit}")
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
    candidates = [
        root / "qdrant" / "qdrant.exe",
        root / "vendor" / "qdrant" / "qdrant.exe",
    ]
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
