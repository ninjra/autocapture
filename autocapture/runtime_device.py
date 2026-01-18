"""Deterministic device selection helpers."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .runtime_env import GpuMode, RuntimeEnvConfig
from .runtime_pause import PauseController


class DeviceKind(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    device_kind: DeviceKind
    torch_device: str | None = None
    ort_providers: list[str] | None = None
    provider_options: dict[str, object] | None = None
    notes: dict[str, object] | None = None


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
    return probe_torch_cuda_available() or probe_ort_cuda_available()


def require_cuda_available(runtime_env: RuntimeEnvConfig) -> None:
    if runtime_env.gpu_mode != GpuMode.ON:
        return
    if not cuda_available():
        raise RuntimeError(
            "GPU_MODE=on but CUDA unavailable; install GPU backend or set GPU_MODE=off."
        )


class DeviceManager:
    def __init__(
        self,
        runtime_env: RuntimeEnvConfig,
        pause_controller: PauseController | None = None,
        *,
        cuda_available_fn: Callable[[], bool] | None = None,
    ) -> None:
        self._env = runtime_env
        self._pause = pause_controller
        self._cuda_available = cuda_available_fn or cuda_available

    def select_device(self) -> DeviceSelection:
        if self._pause and self._pause.is_paused():
            return DeviceSelection(
                device_kind=DeviceKind.CPU,
                torch_device="cpu",
                ort_providers=["CPUExecutionProvider"],
                notes={"reason": "paused"},
            )
        if self._env.gpu_mode == GpuMode.OFF:
            return DeviceSelection(
                device_kind=DeviceKind.CPU,
                torch_device="cpu",
                ort_providers=["CPUExecutionProvider"],
                notes={"reason": "gpu_mode_off"},
            )
        if self._env.gpu_mode == GpuMode.ON:
            if not self._cuda_available():
                raise RuntimeError("CUDA unavailable; install GPU backend or set GPU_MODE=off.")
            return self._cuda_selection()
        if self._cuda_available():
            return self._cuda_selection()
        return DeviceSelection(
            device_kind=DeviceKind.CPU,
            torch_device="cpu",
            ort_providers=["CPUExecutionProvider"],
            notes={"reason": "cuda_unavailable"},
        )

    def _cuda_selection(self) -> DeviceSelection:
        index = max(0, int(self._env.cuda_device_index))
        torch_device = f"cuda:{index}"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return DeviceSelection(
            device_kind=DeviceKind.CUDA,
            torch_device=torch_device,
            ort_providers=providers,
            notes={"cuda_device_index": index},
        )


def backend_config_for(selection: DeviceSelection) -> dict[str, object]:
    if selection.device_kind == DeviceKind.CUDA:
        return {"torch_device": selection.torch_device, "ort_providers": selection.ort_providers}
    return {"torch_device": "cpu", "ort_providers": ["CPUExecutionProvider"]}
