"""Runtime environment switches (GPU mode, profile, pause dir)."""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
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
class RuntimeEnvConfig:
    gpu_mode: GpuMode
    profile: ProfileName
    profile_override: bool
    runtime_dir: Path
    bench_output_dir: Path
    redact_window_titles: bool
    cuda_device_index: int
    cuda_visible_devices: str | None
    schema_version: int = RUNTIME_ENV_SCHEMA_VERSION

    @property
    def pause_latch_path(self) -> Path:
        return self.runtime_dir / "pause.flag"

    @property
    def pause_reason_path(self) -> Path:
        return self.runtime_dir / "pause_reason.json"


def load_runtime_env(env: Mapping[str, str] | None = None) -> RuntimeEnvConfig:
    env = env or os.environ
    logger = logging.getLogger("runtime.env")

    gpu_mode = _parse_enum(
        env.get("GPU_MODE") or env.get("AUTOCAPTURE_GPU_MODE"),
        GpuMode,
        GpuMode.AUTO,
        "GPU_MODE",
        logger,
    )
    profile_value = env.get("PROFILE") or env.get("AUTOCAPTURE_PROFILE")
    profile = _parse_enum(profile_value, ProfileName, ProfileName.FOREGROUND, "PROFILE", logger)
    profile_override = bool(profile_value)

    runtime_dir_raw = env.get("AUTOCAPTURE_RUNTIME_DIR") or env.get("RUNTIME_DIR")
    runtime_dir = (
        Path(runtime_dir_raw).expanduser() if runtime_dir_raw else _default_runtime_dir(logger)
    )
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Failed to create runtime dir %s: %s", runtime_dir, exc)
        if not runtime_dir_raw:
            fallback = Path.home() / ".cache" / "autocapture_runtime"
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

    bench_dir_raw = env.get("AUTOCAPTURE_BENCH_OUTPUT_DIR")
    bench_output_dir = (
        Path(bench_dir_raw).expanduser() if bench_dir_raw else Path("artifacts/bench")
    )

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

    return RuntimeEnvConfig(
        gpu_mode=gpu_mode,
        profile=profile,
        profile_override=profile_override,
        runtime_dir=runtime_dir,
        bench_output_dir=bench_output_dir,
        redact_window_titles=redact_window_titles,
        cuda_device_index=cuda_device_index,
        cuda_visible_devices=cuda_visible_devices,
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
        "redact_window_titles": runtime_env.redact_window_titles,
        "cuda_device_index": runtime_env.cuda_device_index,
        "cuda_visible_devices": runtime_env.cuda_visible_devices,
    }


def _default_runtime_dir(logger: logging.Logger) -> Path:
    if _is_wsl() and Path("/mnt/c").exists():
        return Path("/mnt/c/autocapture_runtime")
    if _is_wsl():
        logger.warning("WSL detected but /mnt/c missing; runtime dir not shared with Windows")
    return Path.home() / ".cache" / "autocapture_runtime"


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    release = platform.release().lower()
    return "microsoft" in release or "wsl" in release


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
