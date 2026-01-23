"""Runtime environment switches (GPU mode, profile, pause dir)."""

from __future__ import annotations

import logging
import os
import platform
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, TypeVar

from .config import AppConfig


RUNTIME_ENV_SCHEMA_VERSION = 1


class GpuMode(str, Enum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"


class ProfileName(str, Enum):
    FOREGROUND = "foreground"
    IDLE = "idle"


@dataclass(frozen=True, slots=True)
class ProfileTuning:
    max_workers: int
    batch_size: int
    poll_interval_ms: int
    max_queue_depth: int
    max_cpu_pct_hint: int

    def as_dict(self) -> dict[str, int]:
        return {
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "poll_interval_ms": self.poll_interval_ms,
            "max_queue_depth": self.max_queue_depth,
            "max_cpu_pct_hint": self.max_cpu_pct_hint,
        }


@dataclass(frozen=True, slots=True)
class RuntimeEnvConfig:
    gpu_mode: GpuMode
    profile: ProfileName
    profile_override: bool
    runtime_dir: Path
    bench_output_dir: Path
    redact_window_titles: bool
    cuda_device_index: int
    cuda_visible_devices: str | None
    log_dir: Path = Path("artifacts/logs")
    pause_latch_override: Path | None = None
    pause_reason_override: Path | None = None
    foreground_tuning: ProfileTuning = field(default_factory=lambda: _default_tuning("foreground"))
    idle_tuning: ProfileTuning = field(default_factory=lambda: _default_tuning("idle"))
    schema_version: int = RUNTIME_ENV_SCHEMA_VERSION

    @property
    def pause_latch_path(self) -> Path:
        if self.pause_latch_override is not None:
            return self.pause_latch_override
        return self.runtime_dir / "pause.flag"

    @property
    def pause_reason_path(self) -> Path:
        if self.pause_reason_override is not None:
            return self.pause_reason_override
        return self.runtime_dir / "pause_reason.json"

    @property
    def bench_dir(self) -> Path:
        return self.bench_output_dir

    def as_dict(self) -> dict[str, object]:
        return runtime_env_snapshot(self)

    @classmethod
    def from_env(cls, cwd: Path | None = None) -> "RuntimeEnvConfig":
        return load_runtime_env(cwd=cwd)


def load_runtime_env(
    env: Mapping[str, str] | None = None, *, cwd: Path | None = None
) -> RuntimeEnvConfig:
    env = env or os.environ
    logger = logging.getLogger("runtime.env")

    gpu_mode = _parse_enum_strict(
        env.get("AUTOCAPTURE_GPU_MODE") or env.get("GPU_MODE"),
        GpuMode,
        GpuMode.AUTO,
        "AUTOCAPTURE_GPU_MODE",
    )
    profile_value = env.get("PROFILE") or env.get("AUTOCAPTURE_PROFILE")
    profile = _parse_enum_strict(
        profile_value, ProfileName, ProfileName.FOREGROUND, "AUTOCAPTURE_PROFILE"
    )
    profile_override = bool(profile_value)

    runtime_dir_raw = env.get("AUTOCAPTURE_RUNTIME_DIR") or env.get("RUNTIME_DIR")
    runtime_dir_raw = _normalize_runtime_path(runtime_dir_raw) if runtime_dir_raw else None
    runtime_dir = (
        Path(runtime_dir_raw).expanduser() if runtime_dir_raw else _default_runtime_dir(logger, cwd)
    )
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Failed to create runtime dir %s: %s", runtime_dir, exc)
        if not runtime_dir_raw:
            fallback = Path.cwd() / ".runtime"
            if fallback != runtime_dir:
                try:
                    fallback.mkdir(parents=True, exist_ok=True)
                    logger.warning("Using fallback runtime dir %s", fallback)
                    runtime_dir = fallback
                except OSError as fallback_exc:
                    logger.warning(
                        "Failed to create fallback runtime dir %s: %s",
                        fallback,
                        fallback_exc,
                    )

    bench_dir_raw = env.get("AUTOCAPTURE_BENCH_DIR") or env.get("AUTOCAPTURE_BENCH_OUTPUT_DIR")
    bench_dir_raw = _normalize_runtime_path(bench_dir_raw) if bench_dir_raw else None
    bench_output_dir = (
        Path(bench_dir_raw).expanduser() if bench_dir_raw else Path("artifacts/bench")
    )
    log_dir_raw = env.get("AUTOCAPTURE_LOG_DIR")
    log_dir_raw = _normalize_runtime_path(log_dir_raw) if log_dir_raw else None
    log_dir = Path(log_dir_raw).expanduser() if log_dir_raw else Path("artifacts/logs")

    pause_latch_raw = env.get("AUTOCAPTURE_PAUSE_LATCH")
    pause_latch_override = (
        Path(_normalize_runtime_path(pause_latch_raw)).expanduser() if pause_latch_raw else None
    )
    pause_reason_raw = env.get("AUTOCAPTURE_PAUSE_REASON")
    pause_reason_override = (
        Path(_normalize_runtime_path(pause_reason_raw)).expanduser() if pause_reason_raw else None
    )

    for path in (bench_output_dir, log_dir):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Failed to create runtime dir %s: %s", path, exc)
    latch_parent = (pause_latch_override or runtime_dir / "pause.flag").parent
    try:
        latch_parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    redact_window_titles = _parse_bool(
        env.get("AUTOCAPTURE_REDACT_WINDOW_TITLES"),
        default=True,
        name="AUTOCAPTURE_REDACT_WINDOW_TITLES",
        logger=logger,
    )

    cuda_device_index = _parse_int(
        env.get("AUTOCAPTURE_CUDA_DEVICE_INDEX"),
        default=0,
        name="AUTOCAPTURE_CUDA_DEVICE_INDEX",
        logger=logger,
    )

    cuda_visible_devices = (
        env.get("AUTOCAPTURE_CUDA_VISIBLE_DEVICES")
        or env.get("CUDA_VISIBLE_DEVICES")
        or str(cuda_device_index)
    )
    if cuda_visible_devices == "":
        cuda_visible_devices = None

    foreground_tuning = _load_profile_tuning(env, "FOREGROUND")
    idle_tuning = _load_profile_tuning(env, "IDLE")

    return RuntimeEnvConfig(
        gpu_mode=gpu_mode,
        profile=profile,
        profile_override=profile_override,
        runtime_dir=runtime_dir,
        bench_output_dir=bench_output_dir,
        redact_window_titles=redact_window_titles,
        cuda_device_index=cuda_device_index,
        cuda_visible_devices=cuda_visible_devices,
        log_dir=log_dir,
        pause_latch_override=pause_latch_override,
        pause_reason_override=pause_reason_override,
        foreground_tuning=foreground_tuning,
        idle_tuning=idle_tuning,
    )


def apply_runtime_env_overrides(
    config: AppConfig,
    runtime_env: RuntimeEnvConfig,
    *,
    logger: logging.Logger | None = None,
) -> AppConfig:
    logger = logger or logging.getLogger("runtime.env")
    if runtime_env.gpu_mode == GpuMode.OFF:
        config.ocr.device = "cpu"
        config.reranker.device = "cpu"
        config.ocr.paddle_ppstructure_use_gpu = False
        logger.info("GPU_MODE=off; forcing CPU for OCR and reranker")
    elif runtime_env.gpu_mode == GpuMode.ON:
        config.ocr.device = "cuda"
        config.reranker.device = "cuda"
        config.reranker.force_cpu_in_active = False
        logger.info("GPU_MODE=on; forcing CUDA for OCR and reranker")
    return config


def configure_cuda_visible_devices(
    runtime_env: RuntimeEnvConfig,
    *,
    logger: logging.Logger | None = None,
) -> None:
    logger = logger or logging.getLogger("runtime.env")
    if runtime_env.gpu_mode == GpuMode.OFF:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("GPU_MODE=off; hiding CUDA devices")
        return
    visible = runtime_env.cuda_visible_devices
    if visible is None:
        return
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
        logger.info("Set CUDA_VISIBLE_DEVICES=%s", visible)
    elif current != visible:
        logger.warning("CUDA_VISIBLE_DEVICES already set (%s); runtime wants %s", current, visible)


def runtime_env_snapshot(runtime_env: RuntimeEnvConfig) -> dict[str, object]:
    return {
        "schema_version": runtime_env.schema_version,
        "gpu_mode": runtime_env.gpu_mode.value,
        "profile": runtime_env.profile.value,
        "profile_override": runtime_env.profile_override,
        "runtime_dir": runtime_env.runtime_dir.as_posix(),
        "bench_output_dir": runtime_env.bench_output_dir.as_posix(),
        "log_dir": runtime_env.log_dir.as_posix(),
        "pause_latch_path": runtime_env.pause_latch_path.as_posix(),
        "pause_reason_path": runtime_env.pause_reason_path.as_posix(),
        "redact_window_titles": runtime_env.redact_window_titles,
        "cuda_device_index": runtime_env.cuda_device_index,
        "cuda_visible_devices": runtime_env.cuda_visible_devices,
        "foreground_tuning": runtime_env.foreground_tuning.as_dict(),
        "idle_tuning": runtime_env.idle_tuning.as_dict(),
    }


def _default_runtime_dir(logger: logging.Logger, cwd: Path | None) -> Path:
    repo_root = _find_repo_root(cwd or Path.cwd())
    if repo_root:
        return repo_root / ".runtime"
    logger.warning("Repo root not found; defaulting runtime dir to ./.runtime")
    return Path.cwd() / ".runtime"


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    release = platform.release().lower()
    return "microsoft" in release or "wsl" in release


def _find_repo_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return None


def _normalize_runtime_path(value: str) -> str:
    raw = value.strip()
    if not raw:
        return raw
    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):[\\\\/](.+)$", raw)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/")
            return f"/mnt/{drive}/{rest}"
        return raw
    match = re.match(r"^/mnt/([A-Za-z])/(.+)$", raw)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2).replace("/", "\\")
        return f"{drive}:\\{rest}"
    return raw


def _parse_bool(raw: str | None, *, default: bool, name: str, logger: logging.Logger) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s=%s; using default %s", name, raw, default)
    return default


def _parse_int(raw: str | None, *, default: int, name: str, logger: logging.Logger) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%s; using default %s", name, raw, default)
        return default


TEnum = TypeVar("TEnum", bound=Enum)


def _parse_enum(
    raw: str | None,
    enum_type: type[TEnum],
    default: TEnum,
    name: str,
    logger: logging.Logger,
) -> TEnum:
    if not raw:
        return default
    normalized = raw.strip().lower()
    for item in enum_type:
        if item.value == normalized:
            return item
    logger.warning("Invalid %s=%s; using default %s", name, raw, default.value)
    return default


def _parse_enum_strict(
    raw: str | None,
    enum_type: type[TEnum],
    default: TEnum,
    name: str,
) -> TEnum:
    if not raw:
        return default
    normalized = raw.strip().lower()
    for item in enum_type:
        if item.value == normalized:
            return item
    allowed = sorted(item.value for item in enum_type)
    raise ValueError(f"{name} must be one of {allowed}; got {raw!r}")


def _default_tuning(name: str) -> ProfileTuning:
    cpu_count = os.cpu_count() or 1
    if name.lower() == "foreground":
        max_workers = max(1, min(4, cpu_count))
        return ProfileTuning(
            max_workers=max_workers,
            batch_size=8,
            poll_interval_ms=100,
            max_queue_depth=1000,
            max_cpu_pct_hint=40,
        )
    return ProfileTuning(
        max_workers=max(1, cpu_count),
        batch_size=32,
        poll_interval_ms=500,
        max_queue_depth=5000,
        max_cpu_pct_hint=20,
    )


def _load_profile_tuning(env: Mapping[str, str], prefix: str) -> ProfileTuning:
    defaults = _default_tuning(prefix.lower())

    def _read_int(name: str) -> int | None:
        raw = env.get(f"AUTOCAPTURE_{prefix}_{name}") or env.get(f"{prefix}_{name}")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{prefix}_{name} must be an integer; got {raw!r}")

    max_workers = _read_int("MAX_WORKERS") or defaults.max_workers
    batch_size = _read_int("BATCH_SIZE") or defaults.batch_size
    poll_interval_ms = _read_int("POLL_INTERVAL_MS") or defaults.poll_interval_ms
    max_queue_depth = _read_int("MAX_QUEUE_DEPTH") or defaults.max_queue_depth
    max_cpu_pct_hint = _read_int("MAX_CPU_PCT_HINT") or defaults.max_cpu_pct_hint

    _validate_positive(max_workers, f"{prefix}_MAX_WORKERS")
    _validate_positive(batch_size, f"{prefix}_BATCH_SIZE")
    _validate_positive(poll_interval_ms, f"{prefix}_POLL_INTERVAL_MS")
    _validate_positive(max_queue_depth, f"{prefix}_MAX_QUEUE_DEPTH")
    _validate_cpu_pct(max_cpu_pct_hint, f"{prefix}_MAX_CPU_PCT_HINT")

    return ProfileTuning(
        max_workers=max_workers,
        batch_size=batch_size,
        poll_interval_ms=poll_interval_ms,
        max_queue_depth=max_queue_depth,
        max_cpu_pct_hint=max_cpu_pct_hint,
    )


def _validate_positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1; got {value}")


def _validate_cpu_pct(value: int, name: str) -> None:
    if value < 1 or value > 100:
        raise ValueError(f"{name} must be between 1 and 100; got {value}")
