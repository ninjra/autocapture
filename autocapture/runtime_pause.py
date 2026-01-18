"""File-based pause latch controller."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PauseState:
    is_paused: bool
    reason: str
    source: str
    updated_at_ms: int
    raw: dict[str, Any] | None = None


class PauseController:
    def __init__(
        self,
        latch_path: Path,
        reason_path: Path | None = None,
        *,
        poll_interval_s: float = 0.5,
        redact_window_titles: bool = True,
    ) -> None:
        self._latch_path = latch_path
        self._reason_path = reason_path
        self._poll_interval_s = max(0.05, poll_interval_s)
        self._redact_titles = redact_window_titles
        if reason_path is not None:
            try:
                reason_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        try:
            latch_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def pause(self, reason: str | dict[str, Any], source: str) -> None:
        self._touch_latch()
        if self._reason_path:
            payload = self._build_reason_payload(reason, source)
            self._atomic_write_json(self._reason_path, payload)

    def resume(self, source: str | None = None) -> None:
        _ = source
        try:
            self._latch_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        if self._reason_path:
            try:
                self._reason_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def is_paused(self) -> bool:
        return self._latch_path.exists()

    def get_state(self) -> PauseState:
        if not self.is_paused():
            return PauseState(False, "", "", 0, None)
        raw = None
        reason = "unknown"
        source = "unknown"
        updated_at_ms = int(time.time() * 1000)
        if self._reason_path and self._reason_path.exists():
            try:
                raw = json.loads(self._reason_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    reason = str(raw.get("reason") or reason)
                    source = str(raw.get("source") or source)
                    ts = raw.get("ts_ms")
                    if isinstance(ts, (int, float)):
                        updated_at_ms = int(ts)
            except Exception:
                raw = None
        return PauseState(True, reason, source, updated_at_ms, raw)

    def wait_until_resumed(
        self,
        timeout: float | None = None,
        *,
        stop_event: Any | None = None,
    ) -> bool:
        start = time.monotonic()
        while self.is_paused():
            if stop_event is not None and getattr(stop_event, "is_set", None):
                try:
                    if stop_event.is_set():
                        return False
                except Exception:
                    pass
            if timeout is not None and time.monotonic() - start >= timeout:
                raise TimeoutError("Pause latch still active")
            time.sleep(self._poll_interval_s)
        return True

    def _touch_latch(self) -> None:
        try:
            fd = os.open(self._latch_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return
        except OSError:
            try:
                self._latch_path.touch(exist_ok=True)
            except OSError:
                return

    def _build_reason_payload(self, reason: str | dict[str, Any], source: str) -> dict[str, Any]:
        now_ms = int(time.time() * 1000)
        if isinstance(reason, dict):
            payload = {"reason": str(reason.get("reason") or "paused"), "source": source}
            for key in ("app", "title", "state", "detail"):
                if key in reason:
                    payload[key] = reason[key]
            if "window_title" in reason and "title" not in payload:
                payload["title"] = reason["window_title"]
        else:
            payload = {"reason": str(reason), "source": source}
        payload["ts_ms"] = now_ms
        if self._redact_titles and "title" in payload:
            payload["title"] = "<redacted>"
        return payload

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        data = json.dumps(payload, sort_keys=True)
        tmp_path.write_text(data, encoding="utf-8")
        os.replace(tmp_path, path)


def paused_guard(pause: PauseController | None, stop_event: Any | None = None) -> bool:
    if pause is None:
        return False
    if not pause.is_paused():
        return False
    pause.wait_until_resumed(stop_event=stop_event)
    if stop_event is None:
        return False
    try:
        return bool(stop_event.is_set())
    except Exception:
        return False
