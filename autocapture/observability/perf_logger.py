"""Structured performance log writer."""

from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path
from typing import Any, Callable

from ..fs_utils import safe_replace
from ..logging_utils import get_logger


class PerfLogger:
    def __init__(
        self,
        data_dir: Path,
        component: str,
        snapshot_fn: Callable[[], dict[str, Any]],
        *,
        interval_s: float = 10.0,
        max_bytes: int = 10_000_000,
        max_files: int = 3,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._component = component
        self._snapshot_fn = snapshot_fn
        self._interval_s = max(1.0, float(interval_s))
        self._max_bytes = max(1_000_000, int(max_bytes))
        self._max_files = max(1, int(max_files))
        self._log = get_logger(f"perf.{component}")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._path = self._data_dir / "perf" / f"{component}.jsonl"

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
            try:
                payload = self._snapshot_fn() or {}
                payload.setdefault("component", self._component)
                payload.setdefault("time_utc", dt.datetime.now(dt.timezone.utc).isoformat())
                self._append(payload)
            except Exception as exc:  # pragma: no cover - defensive
                self._log.debug("Perf log failed: {}", exc)

    def _append(self, payload: dict[str, Any]) -> None:
        path = self._path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                if path.stat().st_size >= self._max_bytes:
                    self._rotate(path)
            except OSError:
                pass
        line = json.dumps(payload, ensure_ascii=True)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            self._log.debug("Perf log write failed: {}", exc)

    def _rotate(self, path: Path) -> None:
        for idx in range(self._max_files - 1, 0, -1):
            src = path.with_name(f"{path.name}.{idx}")
            dst = path.with_name(f"{path.name}.{idx + 1}")
            if src.exists():
                safe_replace(src, dst)
        if path.exists():
            safe_replace(path, path.with_name(f"{path.name}.1"))
