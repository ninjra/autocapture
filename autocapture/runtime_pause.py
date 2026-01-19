"""File-based pause latch controller."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PauseState:
    paused: bool
    reason: str | None
    source: str | None
    ts_ms: int | None
    meta: dict[str, Any] | None = None
    latch_path: str | None = None
    reason_path: str | None = None

    @property
    def is_paused(self) -> bool:
        return self.paused

    @property
    def updated_at_ms(self) -> int:
        return int(self.ts_ms or 0)

    @property
    def raw(self) -> dict[str, Any] | None:
        return self.meta


class PauseController:
    def __init__(
        self,
        latch_path: Path | Any,
        reason_path: Path | None = None,
        *,
        poll_interval_s: float = 0.5,
        redact_window_titles: bool = True,
    ) -> None:
        if not isinstance(latch_path, Path):
            try:
                from .runtime_env import RuntimeEnvConfig  # local import to avoid cycles
            except Exception:
                RuntimeEnvConfig = None
            if RuntimeEnvConfig is not None and isinstance(latch_path, RuntimeEnvConfig):
                runtime_env = latch_path
                latch_path = runtime_env.pause_latch_path
                reason_path = runtime_env.pause_reason_path
                redact_window_titles = runtime_env.redact_window_titles
            else:
                latch_path = Path(latch_path)
        self._latch_path = latch_path
        self._reason_path = reason_path
        self._poll_interval_s = max(0.05, poll_interval_s)
        self._redact_titles = redact_window_titles
        self._log = logging.getLogger("runtime.pause")
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
        self.write_pause(reason, source)

    def resume(self, source: str | None = None) -> None:
        if source is None:
            self._remove_latch()
            self._remove_reason()
            return
        self.clear_pause(source)

    def write_pause(self, reason: str | dict[str, Any], source: str) -> None:
        if self._reason_path:
            payload = self._build_reason_payload(reason, source)
            self._atomic_write_json(self._reason_path, payload)
        self._touch_latch()

    def clear_pause(self, source: str) -> None:
        removed_latch = self._remove_latch()
        if not self._reason_path:
            return
        if not self._reason_path.exists():
            return
        if source:
            try:
                data = json.loads(self._reason_path.read_text(encoding="utf-8"))
            except Exception as exc:
                if removed_latch:
                    self._log.debug("Pause reason unreadable; removing after latch clear: %s", exc)
                    self._remove_reason()
                return
            if isinstance(data, dict):
                existing_source = data.get("source")
                if existing_source and existing_source != source:
                    self._log.info(
                        "Pause reason owned by %s; skipping removal for %s",
                        existing_source,
                        source,
                    )
                    return
        self._remove_reason()

    def is_paused(self) -> bool:
        return self._latch_path.exists()

    def get_state(self) -> PauseState:
        paused = self.is_paused()
        if not paused:
            return PauseState(False, None, None, None, None, str(self._latch_path), None)
        meta: dict[str, Any] | None = None
        reason: str | None = None
        source: str | None = None
        ts_ms: int | None = None
        reason_path = str(self._reason_path) if self._reason_path else None
        if self._reason_path and self._reason_path.exists():
            try:
                raw = json.loads(self._reason_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    reason = str(raw.get("reason") or "") or None
                    source = str(raw.get("source") or "") or None
                    ts = raw.get("ts_ms")
                    if isinstance(ts, (int, float)):
                        ts_ms = int(ts)
                    meta = {k: v for k, v in raw.items() if k not in {"reason", "source", "ts_ms"}}
            except Exception as exc:
                meta = {"error": f"pause_reason_parse:{exc.__class__.__name__}"}
        return PauseState(
            True,
            reason,
            source,
            ts_ms,
            meta,
            str(self._latch_path),
            reason_path,
        )

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
                return False
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
            for key, value in reason.items():
                if key in {"reason", "source", "ts_ms"}:
                    continue
                payload.setdefault(key, value)
        else:
            payload = {"reason": str(reason), "source": source}
        payload["ts_ms"] = now_ms
        if self._redact_titles:
            for key in ("title", "window_title"):
                if key in payload and payload[key]:
                    payload[key] = "<redacted>"
        return payload

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        data = json.dumps(payload, sort_keys=True)
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)

    def _remove_latch(self) -> bool:
        try:
            self._latch_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    def _remove_reason(self) -> None:
        if not self._reason_path:
            return
        try:
            self._reason_path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            return


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


def pause_checkpoint(pause: PauseController, timeout_s: float | None = None) -> None:
    if pause.is_paused():
        pause.wait_until_resumed(timeout=timeout_s)
