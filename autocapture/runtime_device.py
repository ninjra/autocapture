"""Deterministic device selection helpers."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .runtime_env import GpuMode, RuntimeEnvConfig
from .runtime_pause import PauseController


class DeviceKind(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class GpuRequiredError(RuntimeError):
    def __init__(self, message: str, provider_info: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.provider_info = provider_info or {}


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    device_kind: DeviceKind
    compute_device: str
    torch_device: str | None = None
    ort_providers: list[str] | None = None
    provider_options: dict[str, object] | None = None
    notes: dict[str, object] | None = None
    provider_info: dict[str, object] | None = None


def probe_torch_cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def probe_ort_cuda_available() -> bool:
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    try:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        return "CUDAExecutionProvider" in (providers or [])
    except Exception:
        return False


def cuda_available() -> bool:
    available, _ = _detect_cuda()
    return available


def require_cuda_available(runtime_env: RuntimeEnvConfig) -> None:
    if runtime_env.gpu_mode != GpuMode.ON:
        return
    available, info = _detect_cuda()
    if not available:
        raise GpuRequiredError(
            "GPU_MODE=on but CUDA unavailable; install GPU backend or set GPU_MODE=off.",
            info,
        )


class DeviceManager:
    def __init__(
        self,
        runtime_env: RuntimeEnvConfig,
        pause_controller: PauseController | None = None,
        *,
        cuda_available_fn: Callable[[], bool] | None = None,
        cuda_detect_fn: Callable[[], tuple[bool, dict[str, object]]] | None = None,
    ) -> None:
        self._env = runtime_env
        self._pause = pause_controller
        if cuda_detect_fn is not None:
            self._detect_cuda = cuda_detect_fn
        elif cuda_available_fn is not None:
            self._detect_cuda = lambda: (cuda_available_fn(), {"detection_method": "injected"})
        else:
            self._detect_cuda = _detect_cuda

    def select_device(self) -> DeviceSelection:
        compute_device, provider_info = self.resolve_compute_device()
        if compute_device == "cuda":
            return self._cuda_selection(provider_info)
        return DeviceSelection(
            device_kind=DeviceKind.CPU,
            compute_device="cpu",
            torch_device="cpu",
            ort_providers=["CPUExecutionProvider"],
            notes={"reason": provider_info.get("reason", "cpu")},
            provider_info=provider_info,
        )

    def resolve_compute_device(self) -> tuple[str, dict[str, object]]:
        cuda_available, provider_info = self._detect_cuda()
        provider_info = dict(provider_info)
        provider_info.setdefault("cuda_available", cuda_available)
        if self._env.gpu_mode == GpuMode.OFF:
            provider_info.setdefault("reason", "gpu_mode_off")
            return "cpu", provider_info
        if self._env.gpu_mode == GpuMode.ON:
            if not cuda_available:
                raise GpuRequiredError(
                    "GPU_MODE=on but CUDA unavailable; install GPU backend or set GPU_MODE=off.",
                    provider_info,
                )
            return "cuda", provider_info
        if cuda_available:
            return "cuda", provider_info
        provider_info.setdefault("reason", "cuda_unavailable")
        return "cpu", provider_info

    @contextmanager
    def with_gpu_env(self) -> object:
        compute_device, _ = self.resolve_compute_device()
        prior = os.environ.get("CUDA_VISIBLE_DEVICES")
        if compute_device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif prior is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            yield
        finally:
            if prior is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prior

    def _cuda_selection(self, provider_info: dict[str, object]) -> DeviceSelection:
        index = max(0, int(self._env.cuda_device_index))
        torch_device = f"cuda:{index}"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return DeviceSelection(
            device_kind=DeviceKind.CUDA,
            compute_device="cuda",
            torch_device=torch_device,
            ort_providers=providers,
            notes={"cuda_device_index": index},
            provider_info=provider_info,
        )


def backend_config_for(selection: DeviceSelection) -> dict[str, object]:
    if selection.device_kind == DeviceKind.CUDA:
        return {"torch_device": selection.torch_device, "ort_providers": selection.ort_providers}
    return {"torch_device": "cpu", "ort_providers": ["CPUExecutionProvider"]}


def _detect_cuda() -> tuple[bool, dict[str, object]]:
    info: dict[str, object] = {}
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # type: ignore

            available = bool(torch.cuda.is_available())
            info["detection_method"] = "torch"
            info["torch_version"] = getattr(torch, "__version__", None)
            if available:
                try:
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                except Exception:
                    pass
            return available, info
        except Exception as exc:
            info["detection_method"] = "torch"
            info["torch_error"] = f"{exc.__class__.__name__}"
            return False, info
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        info["detection_method"] = "nvidia-smi"
        info["nvidia_smi_ok"] = result.returncode == 0
        if result.returncode == 0 and result.stdout:
            first = result.stdout.splitlines()[0].strip()
            if first:
                info["gpu_name"] = first
        return result.returncode == 0, info
    except FileNotFoundError:
        info["detection_method"] = "none"
        return False, info
    except subprocess.TimeoutExpired:
        info["detection_method"] = "none"
        info["nvidia_smi_ok"] = False
        return False, info
