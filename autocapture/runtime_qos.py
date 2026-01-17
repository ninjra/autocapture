"""Runtime QoS helpers (CPU priority adjustments)."""

from __future__ import annotations

import os

from .logging_utils import get_logger


_log = get_logger("runtime.qos")


def apply_cpu_priority(level: str) -> None:
    normalized = (level or "normal").strip().lower()
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        if os.name == "nt":
            if normalized == "below_normal":
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                proc.nice(psutil.NORMAL_PRIORITY_CLASS)
            return
        if normalized == "below_normal":
            proc.nice(10)
        else:
            proc.nice(0)
    except Exception as exc:  # pragma: no cover - best-effort
        _log.debug("CPU priority update failed: {}", exc)
