"""Heartbeat storage helpers for UX state."""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..fs_utils import fsync_dir, fsync_file, safe_replace, safe_unlink


HEARTBEAT_SCHEMA_VERSION = 1


def write_heartbeat(
    path: Path,
    *,
    component: str,
    status: str,
    signals: dict[str, Any] | None = None,
    errors: list[str] | None = None,
    interval_s: float = 2.0,
) -> None:
    payload = {
        "schema_version": HEARTBEAT_SCHEMA_VERSION,
        "component": component,
        "status": status,
        "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "interval_s": float(interval_s),
        "signals": signals or {},
        "errors": list(errors or []),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp-{path.name}-{int(time.time() * 1000)}")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        fsync_file(tmp_path)
        fsync_dir(tmp_path.parent)
        safe_replace(tmp_path, path)
        fsync_dir(path.parent)
    finally:
        if tmp_path.exists():
            safe_unlink(tmp_path)


def read_heartbeat(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def compute_staleness(
    now: dt.datetime, heartbeat_time: dt.datetime | None, interval_s: float
) -> bool:
    if heartbeat_time is None:
        return True
    if heartbeat_time.tzinfo is None:
        heartbeat_time = heartbeat_time.replace(tzinfo=dt.timezone.utc)
    delta = (now - heartbeat_time).total_seconds()
    threshold = max(10.0, 3.0 * float(interval_s))
    return delta > threshold


class HeartbeatWriter:
    def __init__(
        self,
        path: Path,
        *,
        component: str,
        interval_s: float = 2.0,
        build_payload: Callable[[], tuple[str, dict[str, Any], list[str]]],
    ) -> None:
        self._path = path
        self._component = component
        self._interval_s = float(interval_s)
        self._build_payload = build_payload
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            status, signals, errors = self._build_payload()
            write_heartbeat(
                self._path,
                component=self._component,
                status=status,
                signals=signals,
                errors=errors,
                interval_s=self._interval_s,
            )
