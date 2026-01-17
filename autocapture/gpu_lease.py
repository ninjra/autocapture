"""GPU lease lifecycle helpers."""

from __future__ import annotations

import threading
from typing import Callable

from .logging_utils import get_logger


class GpuLease:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._release_hooks: dict[str, Callable[[str], None]] = {}
        self._log = get_logger("gpu.lease")

    def register_release_hook(self, name: str, hook: Callable[[str], None]) -> None:
        if not name:
            return
        with self._lock:
            self._release_hooks[name] = hook

    def unregister_release_hook(self, name: str) -> None:
        with self._lock:
            self._release_hooks.pop(name, None)

    def release(self, reason: str) -> None:
        hooks: list[tuple[str, Callable[[str], None]]] = []
        with self._lock:
            hooks = list(self._release_hooks.items())
        for name, hook in hooks:
            try:
                hook(reason)
            except Exception as exc:  # pragma: no cover - defensive
                self._log.debug("GPU release hook {} failed: {}", name, exc)


_global_gpu_lease: GpuLease | None = None
_global_lock = threading.Lock()


def get_global_gpu_lease() -> GpuLease:
    global _global_gpu_lease
    with _global_lock:
        if _global_gpu_lease is None:
            _global_gpu_lease = GpuLease()
        return _global_gpu_lease
